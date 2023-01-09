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

/// The energy meter was asked to start sampling while already sampling.
#[derive(Debug, Default)]
pub struct AlreadySampling;

impl From<()> for AlreadySampling {
    fn from(_: ()) -> Self {
        Self
    }
}

#[derive(Default)]
pub struct EnergyReadings {
    /// All current readings in mA
    pub current_readings: Vec<u32>,
    /// All voltage readings in mV
    pub voltage_readings: Vec<u32>,
    pub sample_interval: time::Duration,
}

impl EnergyReadings {
    pub fn new(sample_interval: time::Duration) -> Self {
        Self {
            sample_interval,
            ..Self::default()
        }
    }

    /// Return the cumulative energy consumed in the sampling period. If the energy meter is in use,
    /// then `None` is returned.
    pub fn energy_consumed(&self) -> f64 {
        assert_eq!(self.current_readings.len(), self.voltage_readings.len());

        self.voltage_readings
            .iter()
            .zip(self.current_readings.iter())
            .map(|(&v, &c)| (v * c) as f64 * self.sample_interval.as_secs_f64() / 10e6)
            .sum()
    }

    pub fn write_csv<P: AsRef<path::Path>>(&self, path: P) -> io::Result<()> {
        assert_eq!(self.current_readings.len(), self.voltage_readings.len());

        let mut file = fs::File::create(path)?;

        writeln!(
            file,
            "Time (ms), Voltage (mV), Current (mA), Power (W), Cumulative Energy (J)"
        )?;

        let mut cumulative_energy = 0.0;

        for i in 0..self.current_readings.len() {
            let time = i * (self.sample_interval.as_millis() as usize);
            let voltage = self.voltage_readings[i];
            let current = self.current_readings[i];
            let power = (voltage * current) as f64 / 10e6;
            cumulative_energy += power * self.sample_interval.as_secs_f64();

            writeln!(
                file,
                "{time}, {voltage}, {current}, {power}, {cumulative_energy}"
            )?;
        }

        Ok(())
    }
}

type JEMJoinHandle = thread::JoinHandle<io::Result<(JEMConfig, EnergyReadings)>>;

struct JEMConfig {
    current_fd: fs::File,
    voltage_fd: fs::File,
    sample_interval: time::Duration,
}

enum JEMInner {
    Config(JEMConfig),
    JoinHandle(JEMJoinHandle),
    TempNone,
}

impl JEMInner {
    /// Takes the config if it is a config. Otherwise there is no change
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

    /// Creates a new energy meter with 5ms sampling interval
    pub fn new() -> io::Result<Self> {
        Self::with_sample_interval(time::Duration::from_millis(5))
    }

    pub fn with_sample_interval(sample_interval: time::Duration) -> io::Result<Self> {
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
                sample_interval,
            }),
            should_stop: Arc::new(AtomicBool::new(false)),
        })
    }

    pub fn is_sampling(&self) -> bool {
        matches!(self.inner, JEMInner::JoinHandle(_))
    }

    /// Starts a new thread which collects the energy consumption. Returns an `Err` if energy meter
    /// is already collecting.
    pub fn start_sampling(&mut self) -> Result<(), AlreadySampling> {
        let mut config = self.inner.take_config().ok_or(())?;

        let should_stop = Arc::clone(&self.should_stop);

        // Make sure that should_stop has false in it
        should_stop.store(false, atomic::Ordering::Relaxed);

        self.inner = JEMInner::JoinHandle(std::thread::spawn(move || {
            let mut readings = EnergyReadings::new(config.sample_interval);
            while !should_stop.load(atomic::Ordering::Relaxed) {
                let current_reading = jem_read_value(&mut config.current_fd)?;
                let voltage_reading = jem_read_value(&mut config.voltage_fd)?;

                readings.current_readings.push(current_reading);
                readings.voltage_readings.push(voltage_reading);

                thread::sleep(config.sample_interval);
            }

            Ok((config, readings))
        }));

        Ok(())
    }

    /// Stops the sampling. Returns an error if the energy meter wasn't sampling or there was an
    /// issue with stopping the energy meter.
    pub fn stop_sampling(&mut self) -> io::Result<EnergyReadings> {
        let h = self
            .inner
            .take_join_handle()
            .ok_or_else(|| io::Error::other("Energy meter hasn't started sampling"))?;

        self.should_stop.store(true, atomic::Ordering::Relaxed);

        let (inner, energy_readings) = h
            .join()
            .map_err(|_| io::Error::other("Energy meter panicked while sampling"))??;

        self.inner = JEMInner::Config(inner);

        Ok(energy_readings)
    }
}

fn jem_read_value(file: &mut fs::File) -> io::Result<u32> {
    let mut buf = [0u8; 32];
    let nread = file.read_at(&mut buf, 0)?;
    let s = std::str::from_utf8(&buf[..nread]).map_err(io::Error::other)?;

    // ignore \n at the end
    s[..(nread - 1)].parse().map_err(io::Error::other)
}
