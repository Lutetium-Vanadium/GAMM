use std::{
    fs,
    io::{self, Write},
    os::unix::fs::FileExt,
    path,
    sync::{
        atomic::{self, AtomicBool},
        Arc,
    },
    thread, time,
};

struct JEMConfig {
    current_fd: fs::File,
    voltage_fd: fs::File,
    /// All current readings in mA
    current_readings: Vec<u32>,
    /// All voltage readings in mV
    voltage_readings: Vec<u32>,
    sample_every: time::Duration,
}

type JEMJoinHandle = thread::JoinHandle<io::Result<JEMConfig>>;

enum JEMInner {
    Config(JEMConfig),
    JoinHandle(JEMJoinHandle),
    TempNone,
}

impl JEMInner {
    fn take_config(&mut self) -> Option<JEMConfig> {
        match std::mem::replace(self, JEMInner::TempNone) {
            JEMInner::Config(config) => Some(config),
            this => {
                *self = this;
                None
            }
        }
    }

    /// Takes the join handle if it is a join handle. Otherwise there is no change
    fn take_join_handle(&mut self) -> Option<JEMJoinHandle> {
        match std::mem::replace(self, JEMInner::TempNone) {
            JEMInner::JoinHandle(handle) => Some(handle),
            this => {
                *self = this;
                None
            }
        }
    }

    fn config(&self) -> Option<&JEMConfig> {
        match self {
            JEMInner::Config(config) => Some(config),
            _ => None,
        }
    }
}

pub struct JetsonEnergyMeter {
    should_stop: Arc<AtomicBool>,
    inner: JEMInner,
}

impl Drop for JetsonEnergyMeter {
    fn drop(&mut self) {
        // If the energy meter is dropped before stopping the collection, task the collection to
        // stop as energy data collected can no longer be accessed
        if self.is_sampling() {
            self.should_stop.store(true, atomic::Ordering::Relaxed);
        }
    }
}

impl JetsonEnergyMeter {
    const SYS_BASE: &'static str = "/sys/bus/i2c/drivers/ina3221/1-0040/hwmon";

    pub fn new() -> io::Result<Self> {
        Self::new_sample_every(time::Duration::from_millis(5))
    }

    pub fn new_sample_every(sample_every: time::Duration) -> io::Result<Self> {
        let mut dir = fs::read_dir(Self::SYS_BASE)?;
        let fd_path = dir.next();

        if fd_path.is_none() || dir.count() != 0 {
            return Err(io::Error::other(
                "Multiple directories found, expecting only one",
            ));
        }

        let mut fd_path = fd_path.unwrap()?.path();
        fd_path.push("in2_input");
        let voltage_fd = fs::File::open(&fd_path)?;

        fd_path.pop();
        fd_path.push("curr2_input");
        let current_fd = fs::File::open(&fd_path)?;

        Ok(Self {
            inner: JEMInner::Config(JEMConfig {
                current_fd,
                voltage_fd,
                current_readings: Vec::new(),
                voltage_readings: Vec::new(),
                sample_every,
            }),
            should_stop: Arc::new(AtomicBool::new(false)),
        })
    }

    pub fn is_sampling(&self) -> bool {
        matches!(self.inner, JEMInner::JoinHandle(_))
    }

    /// Starts a new thread which collects the energy consumption. Returns an `Err` if energy meter
    /// is already collecting.
    pub fn start_sampling(&mut self) -> Result<(), ()> {
        let mut config = self.inner.take_config().ok_or(())?;

        let should_stop = Arc::clone(&self.should_stop);

        // Make sure that should_stop has false in it
        should_stop.store(false, atomic::Ordering::Relaxed);
        config.voltage_readings.clear();
        config.current_readings.clear();

        self.inner = JEMInner::JoinHandle(std::thread::spawn(move || {
            while !should_stop.load(atomic::Ordering::Relaxed) {
                let current_reading = jem_read_value(&mut config.current_fd)?;
                let voltage_reading = jem_read_value(&mut config.voltage_fd)?;

                config.current_readings.push(current_reading);
                config.voltage_readings.push(voltage_reading);

                thread::sleep(config.sample_every);
            }

            Ok(config)
        }));

        Ok(())
    }

    /// Stops the sampling. Returns an error if the energy meter wasn't sampling or there was an
    /// issue with stopping the energy meter.
    pub fn stop_sampling(&mut self) -> io::Result<()> {
        let h = self
            .inner
            .take_join_handle()
            .ok_or_else(|| io::Error::other("Energy meter hasn't started sampling"))?;

        self.should_stop.store(true, atomic::Ordering::Relaxed);

        self.inner = JEMInner::Config(
            h.join()
                .map_err(|_| io::Error::other("Energy meter panicked while sampling"))??,
        );
        Ok(())
    }

    /// Return the cumulative energy consumed in the sampling period. If the energy meter is in use,
    /// then `None` is returned.
    pub fn energy_consumed(&self) -> Option<f64> {
        let config = self.inner.config()?;

        assert_eq!(config.current_readings.len(), config.voltage_readings.len());

        Some(
            config
                .voltage_readings
                .iter()
                .zip(config.current_readings.iter())
                .map(|(&v, &c)| (v * c) as f64 * config.sample_every.as_secs_f64() / 10e6)
                .sum(),
        )
    }

    pub fn write_csv<P: AsRef<path::Path>>(&self, path: P) -> io::Result<()> {
        let config = self.inner.config().ok_or_else(|| {
            io::Error::other("Cannot read readings while energy meter is in use.")
        })?;

        let mut file = fs::File::create(path)?;

        writeln!(
            file,
            "Time (ms), Voltage (mV), Current (mA), Power (W), Cumulative Energy (J)"
        )?;

        assert_eq!(config.current_readings.len(), config.voltage_readings.len());

        let mut cumulative_energy = 0.0;

        for i in 0..config.current_readings.len() {
            let time = i * (config.sample_every.as_millis() as usize);
            let voltage = config.voltage_readings[i];
            let current = config.current_readings[i];
            let power = (voltage * current) as f64 / 10e6;
            cumulative_energy += power * config.sample_every.as_secs_f64();

            writeln!(
                file,
                "{}, {}, {}, {}, {}",
                time, voltage, current, power, cumulative_energy
            )?;
        }

        Ok(())
    }
}

fn jem_read_value(file: &mut fs::File) -> io::Result<u32> {
    let mut buf = [0u8; 32];
    let nread = file.read_at(&mut buf, 0)?;
    let s = std::str::from_utf8(&buf[..nread]).map_err(io::Error::other)?;

    // ignore \n at the end
    s[..(nread - 1)].parse().map_err(io::Error::other)
}
