#ifndef NLBRIDGE_ONNX_H
#define NLBRIDGE_ONNX_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct NlbOnnxHandle NlbOnnxHandle;
uint32_t nlb_onnx_abi_version(void);
/* Caller verifies the model bundle before open. Config is UTF-8 JSON.
   On failure, *error_json is owned JSON: free using nlb_onnx_string_free. */
NlbOnnxHandle *nlb_onnx_open(const char *config_json, char **error_json);
/* Both functions return owned JSON; embed accepts a JSON array of 1..32 strings. */
char *nlb_onnx_embed(NlbOnnxHandle *handle, const char *texts_json);
char *nlb_onnx_info(NlbOnnxHandle *handle);
/* Free exactly once, with no concurrent calls using this handle. */
void nlb_onnx_free(NlbOnnxHandle *handle);
void nlb_onnx_string_free(char *text);
#ifdef __cplusplus
}
#endif
#endif
