#include "hash_grid.inc.hlsl"

#define ENABLE_DECAY 1

RWByteAddressBuffer rw_decay_buffer;
RWStructuredBuffer<HashGridCell> rw_storage_buffer;

#define NUM_THREADS_PER_GROUP 64
#define NUM_CELLS_PER_THREAD HASH_GRID_BUCKET_SIZE // 16
#define NUM_CELLS_PER_GROUP (NUM_THREADS_PER_GROUP * NUM_CELLS_PER_THREAD) // 1024

// Each thread processes a bucket
[numthreads(NUM_THREADS_PER_GROUP, 1, 1)]
void main(uint dispatch_id: SV_DispatchThreadID, uint group_id: SV_GroupID, uint group_thread_id: SV_GroupThreadID)
{
    #if !ENABLE_DECAY
    return;
    #endif

    const uint group_addr_offset = group_id * NUM_CELLS_PER_GROUP;
    const uint addr_offset = group_addr_offset + group_thread_id * NUM_CELLS_PER_THREAD;

    // Process by set of 4 cells
    uint bucket_occupantion_mask = 0;
    static const uint NUM_SETS = NUM_CELLS_PER_THREAD / 4;
    uint decal_buffer_local_slice[NUM_SETS]; // store update decay values in local array for later compaction
    uint set = 0; // this counts the number of sets that need to be updated
    for ( ; set < NUM_SETS; )
    {
        // Load 4 decay values (4 bytes)
        uint set_addr = addr_offset + set * 4;
        uint decay_values_packed = rw_decay_buffer.Load(set_addr);

        // NOTE_break: Because each thread process a bucket, and the bucket is always compact, if we found a empty cell, the rest of the bucket must be empty
        if (decay_values_packed == 0)
        {
            break;
        }

        // Update and deallocate cells
        uint decay_values_packed_updated = 0;
        uint decay;
        for (uint i = 0; i < 4; ++i)
        {
            uint shift = i * 8;
            decay = (decay_values_packed >> shift) & 0xFF;
            if (decay == 0)
            {
                // see NOTE_break
                break;
            }
            // TODO adaptive delta value based on delta-time between frames?
            decay = decay - 1;
            if (decay == 0)
            {
                // Deallocate (and reset) the cell
                uint addr = set_addr + i;
                rw_storage_buffer[addr] = HashGridCell::empty();
            }
            else
            {
                // store updated decay value
                decay_values_packed_updated |= (decay << shift);

                // mark the cell as occupied
                bucket_occupantion_mask |= (1u << (set * 4 + i));
            }
        }
        //rw_decay_buffer.Store(set_addr, decay_values_packed_updated);
        decal_buffer_local_slice[set] = decay_values_packed_updated;

        ++set;

        // early out
        if (decay == 0)
        {
            // see NOTE_break
            break;
        }
    }

    // Compact the bucket (storage)
    if (bucket_occupantion_mask != 0)
    {
        while (true)
        {
            uint last_cell_index = firstbithigh(bucket_occupantion_mask);
            uint first_space_index = firstbitlow(~bucket_occupantion_mask);
            // Move the last cell to the first empty space
            if (first_space_index < last_cell_index)
            {
                // Move decay value
                uint src_set = last_cell_index >> 2;
                uint dst_set = first_space_index >> 2;
                // 1. take the value
                uint last_cell_shift = (last_cell_index & 0x3) * 8; // NOTE: `&` may be redundant
                uint decay = decal_buffer_local_slice[src_set] >> last_cell_shift;
                // 2. clear
                decal_buffer_local_slice[src_set] ^= (decay << last_cell_shift);
                // 3. assign to the empty slot
                uint first_space_shift = (first_space_index & 0x3) * 8; // NOTE: `&` may be redundant
                decal_buffer_local_slice[dst_set] |= (decay << first_space_shift);

                // Move cell storage
                // take
                HashGridCell cell = rw_storage_buffer[addr_offset + last_cell_index];
                // assign
                rw_storage_buffer[addr_offset + first_space_index] = cell;
                // clear (the checksum and storage)
                rw_storage_buffer[addr_offset + last_cell_index] = HashGridCell::empty();

                // Update occupation mask
                bucket_occupantion_mask |= (1u << first_space_index);
                bucket_occupantion_mask ^= (1u << last_cell_index);
            }
            else
            {
                // the bucket is compact; done
                break;
            }
        }
    }

    // Update the decay buffer
    for (uint i = 0; i < set; ++i)
    {
        rw_decay_buffer.Store(addr_offset + i * 4, decal_buffer_local_slice[i]);
    }
}