#ifndef NLBRIDGE_H
#define NLBRIDGE_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
uint32_t nlb_abi_version(void);
void *nlb_index_new(const float *data, size_t rows, size_t dim);
void nlb_index_free(void *index);
/* Returns count or a negative error. Arrays out_rows/out_scores hold k elements.
   index is immutable and can be searched concurrently, but not freed concurrently. */
intptr_t nlb_search(const void *index, const float *query, size_t query_len,
                    const uint8_t *mask, size_t mask_len, size_t k,
                    size_t *out_rows, double *out_scores);
/* Returned UTF-8 JSON owns its memory. Always free using nlb_string_free. */
char *nlb_render_json(const char *input);
void nlb_string_free(char *value);
#ifdef __cplusplus
}
#endif
#endif
