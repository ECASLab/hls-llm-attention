#include "hls_math.h"
#include "common/config.h"

// ---------------------------------------------------------------------------------
// --- Kernel config parameters ---
// ---------------------------------------------------------------------------------
constexpr uint32_t QRF = 64; // (Q rows fraction) cantidad de filas de los tiles Q
constexpr uint32_t KRF = 64; // (K rows fraction) cantidad de filas de los tiles K
constexpr uint32_t DKF = 64; // (dk fraction) cantidad de columnas de los tiles Q y K
// VCF must be a multpiple of kPackets!
constexpr uint32_t VCF = 64; // (V columns fraction) cantidad de filas de los tiles V
constexpr uint32_t B_TILES_CACHE_BYTES = 16384; // cantidad de bytes en la cache para guardar tiles B
// Auto computed Kernel config parameters
constexpr uint32_t B_TILES_CACHE_SIZE = B_TILES_CACHE_BYTES / (sizeof(DataT)*QRF*KRF); // cantidad de tiles B que caben en la cache
// ---------------------------------------------------------------------------------
// --- Aux functions ---
// ---------------------------------------------------------------------------------
static inline void unpack_bus_word(RawDataT bus_word, DataT buffer[kPackets])
{
    for (int word_idx = 0; word_idx < kPackets; ++word_idx)
    {
        #pragma HLS UNROLL

        const int lo = word_idx * kDataWidth;
        const int hi = (word_idx + 1) * kDataWidth - 1;

        RawSingleDataT bits = bus_word.range(hi, lo);

        AccT u; GET_RAW(u) = bits;
        buffer[word_idx] = GET_NUMBER(u);
    }
}
static inline RawDataT pack_bus_word(DataT words[kPackets])
{
    #pragma HLS INLINE
    RawDataT bus_word = 0;
    for (int p = 0; p < kPackets; ++p)
    {
        #pragma HLS UNROLL
        #ifdef USE_UNION
            AccT u;
            GET_NUMBER(u) = words[p];
            const RawSingleDataT bits = GET_RAW(u);
        #else
            const RawSingleDataT bits = GET_RAW(words[p]);
        #endif
            const int hi = (p+1)*kDataWidth - 1;
            const int lo = p*kDataWidth;
            bus_word.range(hi, lo) = bits;
    }
    return bus_word;
}
static inline void load_data_to_FIFO(RawDataT *m_axi_pointer, int initial_tile_row, int tile_row_range, int initial_tile_column, int tile_column_range, int total_matrix_rows, int total_matrix_columns, int tile_RF, int tile_CF, StreamT &FIFO)
{
    // If any range is 0, send tile filled with 0s
    if (tile_row_range <= 0 || tile_column_range <= 0)
    {
        const RawDataT zero_word = 0;
        q_rows_zero:
        for (int rel_row = 0; rel_row < tile_RF; ++rel_row)
        {
            q_cols_zero:
            for (int p = 0; p < tile_CF; p += kPackets)
            {
                #pragma HLS PIPELINE II=1

                FIFO.write(zero_word);
            }
        }
        return;
    }

    // Load tile
    DataT buffer[kPackets];
    #pragma HLS ARRAY_PARTITION variable=buffer complete dim=1

    const int total_bus_words = (total_matrix_rows * total_matrix_columns + kPackets - 1) / kPackets;
    RawDataT bus_word_in, next_bus_word_in = 0; // The next_bus_word in case we need to read from two words because of alignment
    q_rows:
    for (int rel_row = 0; rel_row < tile_RF; ++rel_row)
    {
        bool row_in_matrix = (initial_tile_row + rel_row) < total_matrix_rows;
        bool row_in_range = rel_row < tile_row_range;
        bool is_this_row_aligned = ((initial_tile_row + rel_row)*total_matrix_columns + initial_tile_column) % kPackets == 0;

        q_cols: // Move through row in bus_words
        for (int p = 0; p < tile_CF; p += kPackets)
        {
            #pragma HLS PIPELINE II=1

            int starting_elem_idx = (initial_tile_row+rel_row)*total_matrix_columns + initial_tile_column + p;
            int bus_word_idx = starting_elem_idx / kPackets;
            int word_offset = starting_elem_idx % kPackets;

            if(row_in_matrix && row_in_range && bus_word_idx < total_bus_words)
            {
                bool need_to_load_bus_word_in = (!is_this_row_aligned && p==0) || is_this_row_aligned;
                bus_word_in = need_to_load_bus_word_in
                    ? m_axi_pointer[bus_word_idx]
                    : next_bus_word_in;

                bool need_to_load_next_bus_word_in = (!is_this_row_aligned && bus_word_idx+1 < total_bus_words);
                next_bus_word_in = need_to_load_next_bus_word_in
                    ? m_axi_pointer[bus_word_idx+1] // Read next word if previous is not aligned
                    : RawDataT(0);
            }
            
            DataT word_value;
            for (int word_idx = 0; word_idx < kPackets; ++word_idx)
            {
                #pragma HLS UNROLL

                bool word_in_matrix = row_in_matrix && ((initial_tile_column + p + word_idx) < total_matrix_columns);
                bool word_in_range = row_in_range && ((p + word_idx) < tile_column_range);
                bool word_in_bounds = word_in_matrix && word_in_range;

                if (word_in_bounds)
                {
                    int src = word_offset + word_idx;
                    bool use_bus_word_0 = (src < kPackets);
                    int window_idx = use_bus_word_0 ? src : (src - kPackets);

                    const int lo = window_idx * kDataWidth;
                    const int hi = (window_idx + 1) * kDataWidth - 1;

                    RawSingleDataT bits = use_bus_word_0 ? bus_word_in.range(hi, lo)
                                                         : next_bus_word_in.range(hi, lo);
                    AccT u; GET_RAW(u) = bits;
                    word_value = GET_NUMBER(u);
                }
                else
                {
                    word_value = ZERO;
                }
                buffer[word_idx] = word_value;
            }
            RawDataT bus_word_out = pack_bus_word(buffer);
            FIFO.write(bus_word_out);
        }
    }
}
static inline void load_q_tile_from_q_FIFO(StreamT &q_FIFO, DataT q_tile[QRF][DKF])
{
    int bus_words_per_row_tile_q = DKF / kPackets;
    if (DKF % kPackets != 0)
    {
        bus_words_per_row_tile_q++;
    }

    DataT buffer[kPackets];
    #pragma HLS ARRAY_PARTITION variable=buffer complete

    load_q_tile_rows:
    for (uint32_t row=0; row < QRF; ++row)
    {
        load_bus_words_from_q_FIFO:
        for (int bus_word_idx=0; bus_word_idx < bus_words_per_row_tile_q; ++bus_word_idx)
        {
            #pragma HLS PIPELINE II=1
            RawDataT raw_data = q_FIFO.read();
            unpack_bus_word(raw_data, buffer);

            load_q_tile_cols:
            for (int word_idx=0; word_idx < kPackets; ++word_idx)
            {
                #pragma HLS UNROLL
                int col_idx = bus_word_idx*kPackets + word_idx;
                if(col_idx < DKF)
                {
                    q_tile[row][bus_word_idx*kPackets + word_idx] = buffer[word_idx];
                }
            }
        }
    }
}
static inline void load_k_tile_from_k_FIFO(StreamT &k_FIFO, DataT k_tile[KRF][DKF])
{
    int bus_words_per_row_tile_k = DKF / kPackets;
    if (DKF % kPackets != 0)
    {
        bus_words_per_row_tile_k++;
    }

    DataT buffer[kPackets];
    #pragma HLS ARRAY_PARTITION variable=buffer complete

    load_k_tile_rows:
    for (uint32_t row=0; row < KRF; ++row)
    {
        load_bus_words_from_k_FIFO:
        for (int bus_word_idx=0; bus_word_idx < bus_words_per_row_tile_k; ++bus_word_idx)
        {
            #pragma HLS PIPELINE II=1
            RawDataT raw_data = k_FIFO.read();
            unpack_bus_word(raw_data, buffer);

            load_k_tile_cols:
            for (int word_idx=0; word_idx < kPackets; ++word_idx)
            {
                #pragma HLS UNROLL
                int col_idx = bus_word_idx*kPackets + word_idx;
                if(col_idx < DKF)
                {
                    k_tile[row][bus_word_idx*kPackets + word_idx] = buffer[word_idx];
                }
            }
        }
    }
}
static inline void load_v_tile_from_v_FIFO(StreamT &v_FIFO, DataT v_tile[KRF][VCF])
{
    int bus_words_per_row_tile_v = VCF / kPackets;
    if (VCF % kPackets != 0)
    {
        bus_words_per_row_tile_v++;
    }

    DataT buffer[kPackets];
    #pragma HLS ARRAY_PARTITION variable=buffer complete

    load_v_tile_rows:
    for (uint32_t row=0; row < KRF; ++row)
    {
        load_bus_words_from_v_FIFO:
        for (int bus_word_idx=0; bus_word_idx < bus_words_per_row_tile_v; ++bus_word_idx)
        {
            #pragma HLS PIPELINE II=1
            RawDataT raw_data = v_FIFO.read();
            unpack_bus_word(raw_data, buffer);

            load_v_tile_cols:
            for (int word_idx=0; word_idx < kPackets; ++word_idx)
            {
                #pragma HLS UNROLL
                int col_idx = bus_word_idx*kPackets + word_idx;
                if(col_idx < VCF)
                {
                    v_tile[row][bus_word_idx*kPackets + word_idx] = buffer[word_idx];
                }
            }
        }
    }
}
static inline void clear_a_tile(DataT a_tile[QRF][VCF])
{
    clear_a_tile_rows:
    for (int row=0; row < QRF; ++row)
    {
        clear_a_tile_blocks:
        for (int c0=0; c0 < VCF; c0+=kPackets)
        {
            #pragma HLS PIPELINE II=1
            clear_vec:
            for (int u=0; u<kPackets; ++u)
            {
                #pragma HLS UNROLL
                int column = c0 + u;
                if (column < VCF)
                {
                    a_tile[row][column] = ZERO;
                }
            }
        }
    }
}
static inline void accumulate_a_tile(const DataT b_tile[QRF][KRF], const DataT v_tile[KRF][VCF], uint32_t q_range, uint32_t k_range, uint32_t v_range, DataT a_tile[QRF][VCF])
{
acc_rows:
    for (uint32_t q = 0; q < QRF; ++q)
    {
    acc_cols:
        for (uint32_t v = 0; v < VCF; ++v)
        {
            #pragma HLS PIPELINE II=1
            DataT acc = a_tile[q][v];

        acc_depth:
            for (uint32_t k = 0; k < KRF; ++k)
            {
                #pragma HLS UNROLL
                if (q < q_range && v < v_range && k < k_range)
                {
                    acc += b_tile[q][k] * v_tile[k][v];
                }
            }

            if (q < q_range && v < v_range)
            {
                a_tile[q][v] = acc;
            }
        }
    }
}
static inline void normalize_a_tile(DataT a_tile[QRF][VCF], const DataT exps_sum_per_row[QRF], uint32_t q_range, uint32_t v_range)
{
    normalize_rows:
    for (uint32_t row = 0; row < QRF; ++row)
    {
        DataT denom = exps_sum_per_row[row];

        normalize_cols:
        for (uint32_t col = 0; col < VCF; ++col)
        {
            #pragma HLS PIPELINE II=1
            if (row < q_range && col < v_range)
            {
                a_tile[row][col] /= denom;
            }
        }
    }
}
static inline void store_a_tile_to_a_FIFO(const DataT a_tile[QRF][VCF], uint32_t q_range, uint32_t v_range, StreamT &a_FIFO)
{
    DataT buffer[kPackets];
    #pragma HLS ARRAY_PARTITION variable=buffer complete

    store_a_rows:
    for (uint32_t row = 0; row < QRF; ++row)
    {
        store_a_cols:
        for (uint32_t col0 = 0; col0 < VCF; col0 += kPackets)
        {
            #pragma HLS PIPELINE II=1

            for (uint32_t p = 0; p < kPackets; ++p)
            {
                #pragma HLS UNROLL
                uint32_t col = col0 + p;
                if (row < q_range && col < v_range)
                {
                    buffer[p] = a_tile[row][col];
                }
                else
                {
                    buffer[p] = ZERO;
                }
            }

            RawDataT word = pack_bus_word(buffer);
            a_FIFO.write(word);
        }
    }
}
static inline void write_a_tile_to_memory(const DataT a_tile[QRF][VCF], uint32_t q0, uint32_t q_range, uint32_t v0, uint32_t v_range, uint32_t l, uint32_t dv, RawDataT *A)
{
    DataT buffer[kPackets];
    #pragma HLS ARRAY_PARTITION variable=buffer complete

    const uint64_t total_bus_words = (l * dv + kPackets - 1) / kPackets;

    write_rows:
    for (uint32_t row = 0; row < q_range; ++row)
    {
        uint64_t global_row = q0 + row;
        if (global_row >= l) { continue; }

        write_cols:
        for (uint32_t col = 0; col < v_range; ++col)
        {
            #pragma HLS PIPELINE II=1

            uint64_t global_col = v0 + col;
            if (global_col >= dv) { continue; }

            uint32_t elem_idx     = global_row * dv + global_col;
            uint32_t bus_word_idx = elem_idx / kPackets;

            if (bus_word_idx < total_bus_words)
            {
                uint32_t word_offset  = elem_idx % kPackets;
                RawDataT bus_word = A[bus_word_idx];
                unpack_bus_word(bus_word, buffer);
                buffer[word_offset] = a_tile[row][col];
                A[bus_word_idx] = pack_bus_word(buffer);
            }
        }
    }
}
static inline void initialize_QRF_size_array(DataT accumulator[QRF], DataT value)
{
    initialize_accumulator_loop:
    for (int i=0; i < QRF; ++i)
    {
        #pragma HLS UNROLL
        accumulator[i] = value;
    }
}
static inline void normalize_b_tile(DataT b_tile[QRF][KRF], uint32_t q_range, uint32_t k_range, const DataT maxs_per_row[QRF])
{
    subtract_rows:
    for (uint32_t row = 0; row < QRF; ++row)
    {
        DataT row_max = maxs_per_row[row];

        subtract_cols:
        for (uint32_t col = 0; col < KRF; ++col)
        {
            #pragma HLS PIPELINE II=1
            if (row < q_range && col < k_range)
            {
                DataT value = b_tile[row][col] - row_max;
                b_tile[row][col] = hls::exp(value);
            }
        }
    }
}
static inline void normalize_cached_b_tiles(DataT b_tiles_cache[B_TILES_CACHE_SIZE][QRF][KRF], uint32_t b_tiles_cached_amount, uint32_t q_range, uint32_t k_range, const DataT maxs_per_row[QRF])
{
    normalize_cache_tiles:
    for (uint32_t cache_idx = 0; cache_idx < b_tiles_cached_amount; ++cache_idx)
    {
        normalize_b_tile(b_tiles_cache[cache_idx], q_range, k_range, maxs_per_row);
    }
}
static inline void accumulate_exps_sum_per_row(const DataT b_tile[QRF][KRF], uint32_t q_range, uint32_t k_range, DataT exps_sum_per_row[QRF])
{
    accumulate_exps_rows:
    for (uint32_t row = 0; row < QRF; ++row)
    {
        DataT row_sum = ZERO;

        accumulate_exps_cols:
        for (uint32_t col = 0; col < KRF; ++col)
        {
            #pragma HLS PIPELINE II=1
            if (row < q_range && col < k_range)
            {
                row_sum += b_tile[row][col];
            }
        }

        if (row < q_range)
        {
            exps_sum_per_row[row] += row_sum;
        }
    }
}
static inline void copy_b_tile(DataT destination[QRF][KRF], const DataT source[QRF][KRF])
{
    copy_tile_rows:
    for (int r = 0; r < QRF; ++r)
    {
        copy_tile_cols:
        for (int c = 0; c < KRF; ++c)
        {
            #pragma HLS UNROLL
            destination[r][c] = source[r][c];
        }
    }
}
static inline void zeros_b_tile(DataT b_tile[QRF][KRF])
{
    zero_b_tile_rows:
    for (int r = 0; r < QRF; ++r)
    {
        zero_b_tile_cols:
        for (int c = 0; c < KRF; ++c)
        {
            #pragma HLS UNROLL
            b_tile[r][c] = ZERO;
        }
    }
}
static inline void accumulate_b_tile(DataT q_tile[QRF][DKF], DataT k_tile[KRF][DKF], uint32_t q_range, uint32_t k_range, uint32_t d_range, DataT b_tile[QRF][KRF])
{
    accumulate_rows:
    for (uint32_t q_row = 0; q_row < QRF; ++q_row)
    {
        accumulate_cols:
        for (uint32_t k_row = 0; k_row < KRF; ++k_row)
        {
            DataT acc = b_tile[q_row][k_row];

            accumulate_depth:
            for (uint32_t d = 0; d < DKF; ++d)
            {
                #pragma HLS UNROLL
                if (q_row < q_range && k_row < k_range && d < d_range)
                {
                    acc += q_tile[q_row][d] * k_tile[k_row][d];
                }
            }

            b_tile[q_row][k_row] = acc;
        }
    }
}
static inline void scale_b_tile(DataT b_tile[QRF][KRF], uint32_t q_range, uint32_t k_range, DataT scale)
{
    scale_rows:
    for (uint32_t row = 0; row < QRF; ++row)
    {
        scale_cols:
        for (uint32_t col = 0; col < KRF; ++col)
        {
            #pragma HLS PIPELINE II=1
            if (row < q_range && col < k_range)
            {
                b_tile[row][col] *= scale;
            }
        }
    }
}
static inline void get_b_tile_maxs_per_row(const DataT b_tile[QRF][KRF], uint32_t q_range, uint32_t k_range, DataT maxs_per_row[QRF])
{
    row_loop:
    for (uint32_t row = 0; row < QRF; ++row)
    {
        DataT row_max = NEG_INF;

        col_loop:
        for (uint32_t col = 0; col < KRF; ++col)
        {
            #pragma HLS PIPELINE II=1
            if (row < q_range && col < k_range)
            {
                DataT val = b_tile[row][col];
                row_max = (val > row_max) ? val : row_max;
            }
        }
        maxs_per_row[row] = row_max;
    }
}
static inline void get_new_maxs_per_row(const DataT maxs_per_row[QRF], const DataT maxs_per_row_b_tile[QRF], uint32_t q_range, DataT new_maxs_per_row[QRF])
{
    new_maxs_loop:
    for (uint32_t row = 0; row < QRF; ++row)
    {
        #pragma HLS UNROLL
        if (row < q_range)
        {
            DataT a = maxs_per_row[row];
            DataT b = maxs_per_row_b_tile[row];
            new_maxs_per_row[row] = (a > b) ? a : b;
        }
        else
        {
            new_maxs_per_row[row] = ZERO;
        }
    }
}
static inline void get_changed_rows(const DataT new_maxs_per_row[QRF], const DataT maxs_per_row[QRF], uint32_t q_range, bool changed[QRF])
{
    changed_loop:
    for (uint32_t row = 0; row < QRF; ++row)
    {
        #pragma HLS UNROLL
        if (row < q_range)
        {
            changed[row] = (new_maxs_per_row[row] > maxs_per_row[row]);
        }
        else
        {
            changed[row] = false;
        }
    }
}
static inline bool is_there_any_change(const bool changed[QRF], uint32_t q_range)
{
    bool any = false;
    
    any_changed_loop:
    for (uint32_t row = 0; row < QRF; ++row)
    {
        #pragma HLS UNROLL
        if (row < q_range)
        {
            any |= changed[row];
        }
    }
    return any;
}
static inline void get_rescale_factors(const DataT maxs_per_row[QRF], const DataT new_maxs_per_row[QRF], const bool changed[QRF], uint32_t q_range, DataT rescale_factors[QRF])
{
    rescale_loop:
    for (uint32_t row = 0; row < QRF; ++row)
    {
        #pragma HLS UNROLL
        if (row < q_range && changed[row])
        {
            DataT diff = maxs_per_row[row] - new_maxs_per_row[row];
            rescale_factors[row] = hls::exp(diff);
        }
        else
        {
            rescale_factors[row] = DataT(1);   // neutro (en Python no se usa fuera del subset)
        }
    }
}
static inline void apply_rescale_to_exps(DataT exps_sum_per_row[QRF], const DataT rescale_factors[QRF], const bool changed[QRF], uint32_t q_range)
{
    apply_rescale_loop:
    for (uint32_t row = 0; row < QRF; ++row)
    {
        #pragma HLS UNROLL
        if (row < q_range && changed[row])
        {
            exps_sum_per_row[row] *= rescale_factors[row];
        }
    }
}
static inline void apply_rescale_to_a_tile(DataT a_tile[QRF][VCF], const DataT rescale_factors[QRF], const bool changed[QRF], uint32_t q_range, uint32_t v_range)
{
    rescale_rows:
    for (uint32_t row = 0; row < QRF; ++row)
    {
        DataT factor = rescale_factors[row];

        rescale_cols:
        for (uint32_t col = 0; col < VCF; ++col)
        {
            #pragma HLS PIPELINE II=1
            if (row < q_range && col < v_range && changed[row])
            {
                a_tile[row][col] *= factor;
            }
        }
    }
}
static inline void update_maxs_per_row(DataT maxs_per_row[QRF], const DataT new_maxs_per_row[QRF], uint32_t q_range)
{
    update_maxs_loop:
    for (uint32_t row = 0; row < QRF; ++row)
    {
        #pragma HLS UNROLL
        if (row < q_range)
        {
            maxs_per_row[row] = new_maxs_per_row[row];
        }
    }
}
static inline void clear_cache(DataT b_tiles_cache[B_TILES_CACHE_SIZE][QRF][KRF])
{
    clear_cache_i_loop:
    for (int cache_index = 0; cache_index < B_TILES_CACHE_SIZE; ++cache_index)
    {
        clear_cache_rows_loop:
        for (int row = 0; row < QRF; ++row)
        {
            clear_cache_columns_loop:
            for (int c0 = 0; c0 < KRF; c0 += kPackets)
            {
                #pragma HLS PIPELINE II=1
                for (int u = 0; u < kPackets; ++u)
                {
                    #pragma HLS UNROLL
                    int column = c0 + u;
                    if (column < KRF)
                    {
                        b_tiles_cache[cache_index][row][column] = ZERO;
                    }
                }
            }
        }
    }
}
// ---------------------------------------------------------------------------------
// --- Dataflow functions ---
// ---------------------------------------------------------------------------------
static void load_input
(
    // Inputs
    RawDataT *Q,
    RawDataT *K,
    RawDataT *V,
    uint32_t l,
    uint32_t dk,
    uint32_t dv,

    // Outputs
    StreamT &q_FIFO,
    StreamT &k_FIFO,
    StreamT &v_FIFO
)
{
    uint32_t b_tiles_cached_amount = 0;

    for (uint32_t q0=0; q0 < l; q0+=QRF)
    {
        uint32_t q_range = QRF < (l-q0) ? QRF : l-q0; // min(QRF, l-q0)
        b_tiles_cached_amount = 0;

        for (uint32_t v0=0; v0 < dv; v0+=VCF)
        {
            uint32_t v_range = VCF < (dv-v0) ? VCF : dv-v0; // min(VCF, dv-v0)

            for (uint32_t k0=0; k0 < l; k0+=KRF)
            {
                uint32_t k_range = KRF < (l-k0) ? KRF : l-k0; // min(KRF, l-k0)
                
                load_data_to_FIFO(V, k0, k_range, v0, v_range, l, dv, KRF, VCF, v_FIFO);

                uint32_t b_tile_index = k0 / KRF;
                bool cache_hit = (v0 != 0) && (b_tile_index < b_tiles_cached_amount);

                if (!cache_hit)
                {
                    for (uint32_t d0=0; d0 < dk; d0+=DKF)
                    {
                        uint32_t d_range = DKF < (dk-d0) ? DKF : dk-d0; // min(DKF, dk-d0)
                        load_data_to_FIFO(Q, q0, q_range, d0, d_range, l, dk, QRF, DKF, q_FIFO);
                        load_data_to_FIFO(K, k0, k_range, d0, d_range, l, dk, KRF, DKF, k_FIFO);
                    }

                    if (B_TILES_CACHE_SIZE > 0)
                    {
                        if (b_tiles_cached_amount < B_TILES_CACHE_SIZE)
                        {
                            b_tiles_cached_amount += 1;
                        }
                    }
                }
            }
        }
    }
}
static void compute
(
    // Inputs
    StreamT &q_FIFO,
    StreamT &k_FIFO,
    StreamT &v_FIFO,
    uint32_t l,
    uint32_t dk,
    uint32_t dv,

    // Outputs
    StreamT &a_FIFO
)
{
    /*
    Computes the attention matrix (A) without materializing the intermediate matrices B or S.

    Parameters:
    Q: Query matrix                        (l, dk)
    K: Key matrix                          (l, dk)
    V: Value matrix                        (l, dv)
    l: number of rows in Q, K, and V

    lq: number of rows in the Q tile       (QRF, dk)
    lk: number of rows in the K tile       (KRF, dk)

    Intermediate matrices:
    B: (QK^T)/sqrt(d_k) matrix             (l, l)  <- must not be materialized
    S: softmax(B) matrix                   (l, l)  <- must not be materialized

    Output:
    A: Attention matrix S@V                (l, dv)
    */
    #pragma HLS INLINE off
    

    // --- Constants ---
    const DataT scale = DataT(1) / hls::sqrt(DataT(dk)); // scale
    const uint32_t v_loops_amount = (dv + VCF - 1) / VCF;

    // --- Accumulators ---
    DataT maxs_per_row[QRF];
    #pragma HLS ARRAY_PARTITION variable=maxs_per_row complete
    DataT exps_sum_per_row[QRF];
    #pragma HLS ARRAY_PARTITION variable=exps_sum_per_row complete
    DataT a_tile[QRF][VCF];
    #pragma HLS ARRAY_PARTITION variable=a_tile dim=2 factor=kPackets cyclic

    // --- Aux ---
    DataT new_maxs_per_row[QRF];
    #pragma HLS ARRAY_PARTITION variable=new_maxs_per_row complete
    DataT maxs_per_row_b_tile_eff[QRF];
    #pragma HLS ARRAY_PARTITION variable=maxs_per_row_b_tile_eff complete
    bool changed[QRF];
    #pragma HLS ARRAY_PARTITION variable=changed complete
    DataT rescale_factors[QRF];
    #pragma HLS ARRAY_PARTITION variable=rescale_factors complete

    // --- Tiles ---
    DataT q_tile[QRF][DKF];
    #pragma HLS ARRAY_PARTITION variable=q_tile dim=2 factor=kPackets cyclic
    DataT v_tile[KRF][VCF];
    #pragma HLS ARRAY_PARTITION variable=v_tile dim=2 factor=kPackets cyclic
    DataT k_tile[KRF][DKF];
    #pragma HLS ARRAY_PARTITION variable=k_tile dim=2 factor=kPackets cyclic
    DataT b_tile[QRF][KRF];
    #pragma HLS ARRAY_PARTITION variable=b_tile dim=2 factor=kPackets cyclic

    // Cache
    DataT b_tiles_cache[B_TILES_CACHE_SIZE][QRF][KRF];
    #pragma HLS ARRAY_PARTITION variable=b_tiles_cache dim=3 factor=kPackets cyclic
    clear_cache(b_tiles_cache);
    uint32_t b_tiles_cached_amount = 0;
    bool b_tiles_cache_normalized = false;
    bool cache_hit = false;

    q_level_loop: // Loop over Q tiles
    for (uint32_t q0=0; q0 < l; q0+=QRF)
    {
        uint32_t q_range = QRF < (l-q0) ? QRF : l-q0; // min(QRF, l-q0)

        //Reset accumulators
        initialize_QRF_size_array(maxs_per_row, NEG_INF);
        initialize_QRF_size_array(exps_sum_per_row, ZERO);

        // Clear cache
        if (B_TILES_CACHE_SIZE > 0)
        {
            clear_cache(b_tiles_cache);
            b_tiles_cached_amount = 0;
            b_tiles_cache_normalized = false;
        }

        v_level_loop: // Loop over V tiles
        for (uint32_t v0=0; v0 < dv; v0+=VCF)
        {
            uint32_t v_range = VCF < (dv-v0) ? VCF : dv-v0; // min(VCF, dv-v0)
            
            clear_a_tile(a_tile);
         
            k_level_loop: // Loop over K tiles
            for (uint32_t k0=0; k0 < l; k0+=KRF)
            {
                uint32_t k_range = KRF < (l-k0) ? KRF : l-k0; // min(KRF, l-k0)

                // Load V tile
                load_v_tile_from_v_FIFO(v_FIFO, v_tile);

                int b_tile_idx = k0 / KRF;
                cache_hit = (v0 != 0) && (b_tile_idx < b_tiles_cached_amount);
                if (cache_hit)
                {
                    if((v_loops_amount > 2) && (!b_tiles_cache_normalized))
                    {
                        normalize_cached_b_tiles(b_tiles_cache, b_tiles_cached_amount, q_range, k_range, maxs_per_row);
                        b_tiles_cache_normalized = true;
                    }
                    copy_b_tile(b_tile, b_tiles_cache[b_tile_idx]);
                }
                else
                {
                    zeros_b_tile(b_tile);

                    for (uint32_t d0=0; d0 < dk; d0+=DKF)
                    {
                        uint32_t d_range = DKF < (dk-d0) ? DKF : dk-d0; // min(DKF, dk-d0)
                        load_q_tile_from_q_FIFO(q_FIFO, q_tile);
                        load_k_tile_from_k_FIFO(k_FIFO, k_tile);
                        accumulate_b_tile(q_tile, k_tile, q_range, k_range, d_range, b_tile);
                    }
                    
                    scale_b_tile(b_tile, q_range, k_range, scale);

                    if (v0==0)
                    {
                        get_b_tile_maxs_per_row(b_tile, q_range, k_range, maxs_per_row_b_tile_eff);
                        get_new_maxs_per_row(maxs_per_row, maxs_per_row_b_tile_eff, q_range, new_maxs_per_row);
                        get_changed_rows(new_maxs_per_row, maxs_per_row, q_range, changed);

                        if (is_there_any_change(changed, q_range))
                        {
                            get_rescale_factors(maxs_per_row, new_maxs_per_row, changed, q_range, rescale_factors);
                            apply_rescale_to_exps(exps_sum_per_row, rescale_factors, changed, q_range);
                            apply_rescale_to_a_tile(a_tile, rescale_factors, changed, q_range, v_range);
                            update_maxs_per_row(maxs_per_row, new_maxs_per_row, q_range);
                        }
                    }

                    // --- Caching ---
                    if (B_TILES_CACHE_SIZE > 0 && b_tiles_cached_amount < B_TILES_CACHE_SIZE)
                    {
                        copy_b_tile(b_tiles_cache[b_tile_idx], b_tile);
                        b_tiles_cached_amount++;
                    }
                }

                // --- B tile normalization ---
                if ((cache_hit && !b_tiles_cache_normalized) || !cache_hit)
                {
                    normalize_b_tile(b_tile, q_range, k_range, maxs_per_row);
                }

                if (v0==0)
                {
                    accumulate_exps_sum_per_row(b_tile, q_range, k_range, exps_sum_per_row);
                }
                accumulate_a_tile(b_tile, v_tile, q_range, k_range, v_range, a_tile);
            }
            normalize_a_tile(a_tile, exps_sum_per_row, q_range, v_range);
            store_a_tile_to_a_FIFO(a_tile, q_range, v_range, a_FIFO);
        }
    }
}
static void store_result
(
    // Inputs
    StreamT &a_FIFO,
    uint32_t l,
    uint32_t dv,

    // Outputs
    RawDataT *A
)
{
    const uint32_t bus_words_per_row = (VCF + kPackets - 1) / kPackets;

    DataT a_tile[QRF][VCF];
    #pragma HLS ARRAY_PARTITION variable=a_tile dim=2 factor=kPackets cyclic
    DataT buffer[kPackets];
    #pragma HLS ARRAY_PARTITION variable=buffer complete

    q_level_loop:
    for (uint32_t q0 = 0; q0 < l; q0 += QRF)
    {
        uint32_t q_range = (QRF < (l - q0)) ? QRF : (l - q0);

        v_level_loop:
        for (uint32_t v0 = 0; v0 < dv; v0 += VCF)
        {
            uint32_t v_range = (VCF < (dv - v0)) ? VCF : (dv - v0);

            read_a_rows:
            for (uint32_t row = 0; row < QRF; ++row)
            {
                read_a_words:
                for (uint32_t bus_word_idx = 0; bus_word_idx < bus_words_per_row; ++bus_word_idx)
                {
                    #pragma HLS PIPELINE II=1

                    RawDataT raw_word = a_FIFO.read();
                    unpack_bus_word(raw_word, buffer);

                    read_a_cols:
                    for (uint32_t word_idx = 0; word_idx < kPackets; ++word_idx)
                    {
                        #pragma HLS UNROLL

                        uint32_t col = bus_word_idx * kPackets + word_idx;
                        if (col < VCF)
                        {
                            a_tile[row][col] = buffer[word_idx];
                        }
                    }
                }
            }
            write_a_tile_to_memory(a_tile, q0, q_range, v0, v_range, l, dv, A);
        }
    }
}
// ---------------------------------------------------------------------------------
// --- Top ---
// ---------------------------------------------------------------------------------
extern "C"
{
    void sdpa
    (
        // Inputs
        RawDataT *Q,
        RawDataT *K,
        RawDataT *V,
        uint32_t l,
        uint32_t dk,
        uint32_t dv,

        // Output
        RawDataT *A
    )
    {
        // ---- Interfaces ----
        // #pragma HLS INTERFACE m_axi port=Q bundle=gmem0 offset=slave
        // #pragma HLS INTERFACE m_axi port=K bundle=gmem1 offset=slave
        // #pragma HLS INTERFACE m_axi port=V bundle=gmem2 offset=slave
        // #pragma HLS INTERFACE m_axi port=A bundle=gmem3 offset=slave

        #pragma HLS INTERFACE m_axi     port=Q bundle=gmem0 offset=slave
        #pragma HLS INTERFACE s_axilite port=Q bundle=control
        #pragma HLS INTERFACE m_axi     port=K bundle=gmem1 offset=slave
        #pragma HLS INTERFACE s_axilite port=K bundle=control
        #pragma HLS INTERFACE m_axi     port=V bundle=gmem2 offset=slave
        #pragma HLS INTERFACE s_axilite port=V bundle=control
        #pragma HLS INTERFACE m_axi     port=A bundle=gmem3 offset=slave
        #pragma HLS INTERFACE s_axilite port=A bundle=control

        #pragma HLS INTERFACE s_axilite port=l  bundle=control
        #pragma HLS INTERFACE s_axilite port=dk bundle=control
        #pragma HLS INTERFACE s_axilite port=dv bundle=control
        #pragma HLS INTERFACE s_axilite port=return bundle=control

        static StreamT q_FIFO;
        static StreamT k_FIFO;
        static StreamT v_FIFO;
        static StreamT a_FIFO;
        #pragma HLS stream variable = q_FIFO depth = 32
        #pragma HLS stream variable = k_FIFO depth = 32
        #pragma HLS stream variable = v_FIFO depth = 32
        #pragma HLS stream variable = a_FIFO depth = 32

        #pragma HLS dataflow
        load_input(Q, K, V, l, dk, dv, q_FIFO, k_FIFO, v_FIFO);
        compute(q_FIFO, k_FIFO, v_FIFO, l, dk, dv, a_FIFO);
        store_result(a_FIFO, l, dv, A);
    }
}
