use anyhow::Result;
use csm_rs::{Generator, GeneratorArgs};
use std::sync::Arc;
use tokio::sync::{Mutex, OwnedSemaphorePermit, Semaphore};

/// Pool of pre-loaded `Generator` instances. CSM-1B is autoregressive with a
/// per-session KV cache, so each WebSocket holds an instance for its lifetime.
///
/// `checkout` blocks until an instance is free, returning a guard that
/// releases the instance back to the pool on drop.
pub struct GeneratorPool {
    instances: Vec<Arc<Mutex<Generator>>>,
    sem: Arc<Semaphore>,
    /// Free indices, popped on checkout, pushed on drop.
    free: Arc<Mutex<Vec<usize>>>,
}

impl GeneratorPool {
    pub async fn new(size: usize, args: GeneratorArgs) -> Result<Self> {
        assert!(size >= 1, "pool size must be >= 1");
        let mut instances = Vec::with_capacity(size);
        for i in 0..size {
            log::info!("Loading generator instance {}/{}", i + 1, size);
            let gen = Generator::new(args.clone()).await?;
            instances.push(Arc::new(Mutex::new(gen)));
        }
        let free = (0..size).collect::<Vec<_>>();
        Ok(Self {
            instances,
            sem: Arc::new(Semaphore::new(size)),
            free: Arc::new(Mutex::new(free)),
        })
    }

    pub async fn checkout(&self) -> CheckoutGuard {
        let permit = self.sem.clone().acquire_owned().await.expect("semaphore closed");
        let idx = {
            let mut free = self.free.lock().await;
            free.pop().expect("free list out of sync with semaphore")
        };
        CheckoutGuard {
            generator: self.instances[idx].clone(),
            idx,
            free: self.free.clone(),
            _permit: permit,
        }
    }
}

pub struct CheckoutGuard {
    pub generator: Arc<Mutex<Generator>>,
    idx: usize,
    free: Arc<Mutex<Vec<usize>>>,
    _permit: OwnedSemaphorePermit,
}

impl Drop for CheckoutGuard {
    fn drop(&mut self) {
        // Push the index back. Try the fast path (uncontested try_lock); if
        // that fails, spawn a task to wait. The semaphore permit is dropped
        // after this returns, so a momentary mismatch is fine — never a leak.
        let free = self.free.clone();
        let idx = self.idx;
        let pushed = match free.try_lock() {
            Ok(mut g) => {
                g.push(idx);
                true
            }
            Err(_) => false,
        };
        if !pushed {
            tokio::spawn(async move {
                free.lock().await.push(idx);
            });
        }
    }
}
