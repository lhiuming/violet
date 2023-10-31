/*
 * Compact the selected query indices into a buffer.
 * The selection buffer is mapped by each cell, thus is typically quite sparse.
 */

#include "hash_grid.inc.hlsl"

StructuredBuffer<uint> hash_grid_query_selection_buffer;
RWStructuredBuffer<uint> rw_compact_query_index_buffer;
RWStructuredBuffer<uint> rw_compact_query_cell_addr_buffer;
RWStructuredBuffer<uint> rw_compact_query_counter_buffer;

#define NUM_CELLS_PER_GROUP 128

#define QUERY_INDEX_MASK ( (1 << HASH_GRID_MAX_NUM_QUERIES_BITSHIFT) - 1 )

groupshared uint gs_query_count;
groupshared uint gs_query_index_array[NUM_CELLS_PER_GROUP];
groupshared uint gs_query_cell_addr_array[NUM_CELLS_PER_GROUP];
groupshared uint gs_group_store_pos;

// 1 cell per thread
[numthreads(NUM_CELLS_PER_GROUP, 1, 1)]
void main(uint group_id : SV_GroupID, uint local_id : SV_GroupThreadID) 
{
    if (local_id == 0)
    {
        gs_query_count = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    // Read sparse data (query index) to LDS
    uint group_cell_addr_offset = group_id * NUM_CELLS_PER_GROUP;
    uint cell_addr = group_cell_addr_offset + local_id;
    uint selection_work = hash_grid_query_selection_buffer[cell_addr];
    if (selection_work != 0)
    {
        uint gs_store_pos;
        InterlockedAdd(gs_query_count, 1, gs_store_pos);

        uint query_index = selection_work & QUERY_INDEX_MASK;
        gs_query_index_array[gs_store_pos] = query_index;
        gs_query_cell_addr_array[gs_store_pos] = cell_addr;
    }
    GroupMemoryBarrierWithGroupSync();

    // Allocate global buffer slice for whole group
    if (local_id == 0)
    {
        uint global_store_pos;
        InterlockedAdd(rw_compact_query_counter_buffer[0], gs_query_count, global_store_pos);
        gs_group_store_pos = global_store_pos;
    }
    GroupMemoryBarrierWithGroupSync();

    // Copy LDS to global buffer
    if (local_id < gs_query_count)
    {
        uint store_pos = gs_group_store_pos + local_id;
        rw_compact_query_index_buffer[store_pos] = gs_query_index_array[local_id];
        rw_compact_query_cell_addr_buffer[store_pos] = gs_query_cell_addr_array[local_id];
    }
}