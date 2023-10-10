use std::collections::VecDeque;

use crate::render_device::RenderDevice;
use ash::vk;

pub struct Query(pub u32);

// A bunch of queries
pub struct QueryBatch {
    pool: vk::QueryPool,
    first_query: Query,
    query_count: u32,
}

impl QueryBatch {
    pub fn pool(&self) -> vk::QueryPool {
        self.pool
    }

    pub fn query(&self, index: u32) -> Query {
        assert!(index < self.query_count);
        Query(self.first_query.0 + index)
    }

    pub fn size(&self) -> u32 {
        self.query_count
    }
}

// Allocating continuous queries by a block.
pub struct QueryPool {
    query_pool: vk::QueryPool,
    block_size: u32,
    allocated_bitmask: u64, // a bit per block
}

impl QueryPool {
    pub fn new(rd: &RenderDevice) -> Self {
        let query_count = 512;
        let block_size = (query_count + 63) / 64; // using a 64-bit bitmask
        let query_count = block_size * 64;

        let create_info = vk::QueryPoolCreateInfo::builder()
            .query_count(query_count)
            .query_type(vk::QueryType::TIMESTAMP);
        let query_pool = unsafe { rd.device.create_query_pool(&create_info, None).unwrap() };

        Self {
            query_pool,
            block_size,
            allocated_bitmask: 0,
        }
    }

    pub fn alloc(&mut self, query_count: u32) -> Option<QueryBatch> {
        let num_blocks = (query_count + self.block_size - 1) / self.block_size;

        let mut offset = 0;
        let mut bitmask = self.allocated_bitmask;
        loop {
            let num_zeros = bitmask.trailing_zeros();
            // if we have long engough continuous space
            if num_blocks <= num_zeros {
                break;
            }
            // skip to next space
            else {
                // skip current space
                offset += num_zeros;
                bitmask = bitmask >> num_zeros;
                // skip next non-space
                let num_ones = bitmask.trailing_ones();
                bitmask = bitmask >> num_ones;
                offset += num_ones;
            }

            if num_blocks > 64 - offset {
                // run out of space (or fragmentation)
                return None;
            }
        }

        // mark allocate
        let mask = ((1 << num_blocks) - 1) << offset;
        assert!(self.allocated_bitmask & mask == 0);
        self.allocated_bitmask |= mask;

        let first_index = offset * self.block_size;
        Some(QueryBatch {
            pool: self.query_pool,
            first_query: Query(first_index),
            query_count: num_blocks * self.block_size,
        })
    }

    pub fn release(&mut self, queries: QueryBatch) {
        let first_index = queries.first_query.0;
        let offset = first_index / self.block_size;
        let num_blocks = queries.query_count / self.block_size;
        let mask = ((1 << num_blocks) - 1) << offset;
        assert!(self.allocated_bitmask & mask == mask);
        self.allocated_bitmask ^= mask;
    }
}

struct ProfilingEntry {
    name: String,
    records_ns: VecDeque<u64>, // recent 16 records
    sum_ns: u64,
}

impl ProfilingEntry {
    fn new(name: String) -> Self {
        Self {
            name,
            records_ns: VecDeque::with_capacity(16),
            sum_ns: 0,
        }
    }

    fn add_record_ns(&mut self, time_ns: u64) {
        if self.records_ns.len() == self.records_ns.capacity() {
            let front = self.records_ns.pop_front();
            self.sum_ns -= front.unwrap();
        }
        self.records_ns.push_back(time_ns);
        self.sum_ns += time_ns;
    }

    fn avg_time_ns(&self) -> f64 {
        if self.records_ns.is_empty() {
            0.0
        } else {
            self.sum_ns as f64 / self.records_ns.len() as f64
        }
    }
}

// Associating a batch of queries with profiling entries (names)
struct TimerBatch {
    queries: QueryBatch,
    // entry id, beg query index, end query index
    timers: Vec<(usize, u32, u32)>,
}

pub struct NamedProfiling {
    timer_pool: QueryPool,
    entries: Vec<ProfilingEntry>,
    pending_queries: Vec<TimerBatch>,
    current_batch: Option<TimerBatch>,
}

// ref: https://github.com/KhronosGroup/Vulkan-Samples/tree/main/samples/api/timestamp_queries
impl NamedProfiling {
    pub fn new(rd: &RenderDevice) -> Self {
        Self {
            timer_pool: QueryPool::new(rd),
            entries: Vec::new(),
            pending_queries: Vec::new(),
            current_batch: None,
        }
    }

    fn register_entry(&mut self, name: &str) -> usize {
        for i in 0..self.entries.len() {
            if self.entries[i].name == name {
                return i;
            }
        }

        self.entries.push(ProfilingEntry::new(name.to_string()));
        return self.entries.len() - 1;
    }

    // Start a new batch; you are supposed to reset all queries in it.
    pub fn new_batch(&mut self, timer_capacity: u32) -> &QueryBatch {
        let capacity = timer_capacity * 2;
        let new_batch = TimerBatch {
            queries: self.timer_pool.alloc(capacity).unwrap(),
            timers: Vec::new(),
        };

        if let Some(batch) = self.current_batch.replace(new_batch) {
            self.pending_queries.push(batch);
        }

        &self.current_batch.as_ref().unwrap().queries
    }

    fn finish_batch(&mut self) {
        if let Some(batch) = self.current_batch.take() {
            self.pending_queries.push(batch);
        }
    }

    pub fn new_timer(&mut self, name: &str) -> (Query, Query) {
        let entry_id = self.register_entry(name);

        let batch = self.current_batch.as_mut().unwrap();
        let next_index = (batch.timers.len() as u32) * 2;
        assert!(batch.queries.size() > next_index);

        let beg_index = next_index;
        let end_index = next_index + 1;

        let beg_query = batch.queries.query(next_index);
        let end_query = batch.queries.query(next_index + 1);

        batch.timers.push((entry_id, beg_index, end_index));

        (beg_query, end_query)
    }

    pub fn update(&mut self, rd: &RenderDevice) {
        self.finish_batch();

        // tick period in nanoseconds
        let period = rd.timestamp_period() as f64;

        let mut keep_pending_queries = Vec::new();
        let mut data_buffer = Vec::<[u64; 2]>::new(); // reused buffer
        for batch in self.pending_queries.drain(0..) {
            // query fot the batch
            let first_query = batch.queries.query(0).0;
            let query_count = batch.timers.len() as u32 * 2;
            data_buffer.resize(query_count as usize, [0u64; 2]);
            let vk_not_ready = unsafe {
                match rd.device.get_query_pool_results(
                    batch.queries.pool,
                    first_query,
                    query_count,
                    data_buffer.as_mut_slice(),
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WITH_AVAILABILITY,
                ) {
                    Ok(_) => false,
                    Err(code) => match code {
                        vk::Result::NOT_READY => true,
                        _ => {
                            code.result().unwrap();
                            false
                        }
                    },
                }
            };

            // we treat vk::Result::NOT_READY as a fast path (to reject).
            // " vkGetQueryPoolResults **may** return VK_NOT_READY if there are queries in the unavailable state "
            let fully_available = !vk_not_ready && data_buffer.iter().all(|pair| pair[1] > 0);

            // read results only if every query is available
            if fully_available {
                for (entry_id, beg_index, end_index) in batch.timers {
                    // Read the timestamp
                    let beg_tick = data_buffer[beg_index as usize][0];
                    let end_tick = data_buffer[end_index as usize][0];
                    assert!(end_tick >= beg_tick);
                    // Accumulate to the entry
                    let timespan_ns = ((end_tick - beg_tick) as f64) * period;
                    self.entries[entry_id].add_record_ns(timespan_ns as u64);
                }
                // put back to pool
                self.timer_pool.release(batch.queries);
            } else {
                keep_pending_queries.push(batch);
            }
        }

        self.pending_queries = keep_pending_queries;
    }

    pub fn print(&self) {
        let count = self.entries.len();
        let width = self.entries.iter().map(|e| e.name.len()).max().unwrap_or(0);
        let width = (width + 3) / 4 * 4;
        println!("GPU Profiling: ({} entries)", count);
        for entry in self.entries.iter() {
            let avg_time_ms = entry.avg_time_ns() / 1000_000.0;
            println!("\t{:>width$}: {:.4}ms", entry.name, avg_time_ms);
        }
    }
}
