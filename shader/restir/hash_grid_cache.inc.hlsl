#pragma once

#define HASH_GRID_CAPACITY 65536
#define HASH_GRID_CAPACITY_MASK 0xFFFF

#define HASH_GRID_BUCKET_SIZE 16
#define HASH_GRID_BUCKET_SIZE_BITSHIFT 4

#define HASH_GRID_BASE_CELL_SIZE 0.25

struct QueryRecord
{
    float3 pos;
    float3 dir_in;

    uint4 encode()
    {
        uint3 pos_u32 = asuint(pos);
        return uint4(pos_u32, 0);
    }

    static QueryRecord decode(uint4 data)
    {
        float3 pos = asfloat(data.xyz);
        QueryRecord ret = { pos, float3(0, 0, 0) };
        return ret;
    }
};

uint hash_grid_cell_lod(float3 pos, float3 camera_pos)
{
    float3 delta = pos - camera_pos;
    float dist_sq = dot(delta, delta);
    float lod = max(0.5 * log2(dist_sq) + log2(4.0), 0.0);
    return uint(lod);
}

struct VertexDescriptor
{
    float3 pos; // hit point
    float3 dir_in; // incoming directin / "view" direction for hit point
    uint lod;

    // Hash used to access grid cell
    uint hash()
    {
        // TODO adaptive cell size
        float cell_size = HASH_GRID_BASE_CELL_SIZE * float(1u << lod);

        float3 pos_quant = floor(pos / cell_size);

        // hashing
        // TODO better hash function
        uint3 pos_u32 = asuint(pos_quant);
        uint hash = pos_u32.x ^ pos_u32.y ^ pos_u32.z;
        return hash;
    }

    // Hash used for linear probing
    uint checksum()
    {
        // TODO adaptive cell size
        float cell_size = HASH_GRID_BASE_CELL_SIZE * float(1u << lod);

        float3 pos_quant = floor(pos / cell_size);

        // hashing
        // TODO better hash function
        uint3 pos_u32 = asuint(pos_quant);
        uint hash = pos_u32.x ^ pos_u32.y ^ pos_u32.z;
        hash = hash ^ (hash >> 16);
        // NOTE: zero is reserved to indicate empty cell
        return hash | 0x1;
    }
};

// TODO break checksum into separate buffer, to reduce cache trashing?
struct HashGridCell
{
    uint checksum; // "hash2" / "verification hash"
    float3 radiance; 
};

bool hash_grid_find(StructuredBuffer<HashGridCell> buffer, uint hash, uint checksum, out HashGridCell cell)
{
    // linear probing in the bucket
    uint addr_beg = (hash & HASH_GRID_CAPACITY_MASK) << HASH_GRID_BUCKET_SIZE_BITSHIFT;
    uint addr_end = addr_beg + HASH_GRID_BUCKET_SIZE;
    for (uint addr = addr_beg; addr < addr_end; ++addr)
    {
        cell = buffer[addr];
        if (cell.checksum == checksum)
        {
            return true;
        }
        if (cell.checksum == 0)
        {
            // early out? the bucket need to be compact
            break;
        }
    }

    return false;
}

bool hash_grid_find_or_insert(RWStructuredBuffer<HashGridCell> buffer, uint hash, uint checksum, out uint addr)
{
    // linear probing in the bucket
    uint addr_beg = (hash & HASH_GRID_CAPACITY_MASK) << HASH_GRID_BUCKET_SIZE_BITSHIFT;
    uint addr_end = addr_beg + HASH_GRID_BUCKET_SIZE;
    addr = addr_beg;
    for ( ; addr < addr_end; ++addr)
    {
        uint slot_checksum = buffer[addr].checksum;
        if (slot_checksum == checksum)
        {
            // found existing cell
            return true;
        }
        if (slot_checksum == 0)
        {
            // found an empty slot in the bucket
            // try to insert the cell
            InterlockedCompareExchange(buffer[addr].checksum, 0, checksum, slot_checksum);
            // NOTE: another thread may have inserted the cell with a same checksum
            bool insert_by_this = (slot_checksum == 0);
            bool insert_by_other = (slot_checksum == checksum);
            if ( insert_by_this || insert_by_other )
            {
                // insert successfully (maybe by another thread)
                return true;
            }
        }
    }

    return false;
}