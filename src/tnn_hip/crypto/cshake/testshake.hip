#include "cshake256.h"
#include <cstdlib>
#include <hip/hip_runtime.h> 

// Host function to launch the kernel
void test_cshake256() {
    // // Example input data
    // const char *message = "Test message for cSHAKE256";
    // const char *custom_str = "ProofOfWorkHash";
    // int message_len = strlen(message);
    // int custom_len = strlen(custom_str);
    // int output_len = 32;  // Output length for SHAKE256 (you can adjust this)

    // // Allocate memory on the host
    // unsigned char *output_host = (unsigned char *)malloc(output_len);

    // // Allocate memory on the device
    // char *message_dev, *custom_str_dev;
    // unsigned char *output_dev;

    // printf("message len = %d\ncustom len = %d\n", message_len, custom_len);

    // hipMalloc((void**)&message_dev, message_len * sizeof(char));
    // hipMalloc((void**)&custom_str_dev, custom_len * sizeof(char));
    // hipMalloc((void**)&output_dev, output_len * sizeof(unsigned char));

    // // Copy input data to device
    // hipMemcpy(message_dev, message, message_len * sizeof(char), hipMemcpyHostToDevice);
    // hipMemcpy(custom_str_dev, custom_str, custom_len * sizeof(char), hipMemcpyHostToDevice);

    // // Launch the kernel with 1 block and 1 thread
    // cshake256_single(message_dev, message_len, custom_str_dev, custom_len, output_dev, output_len);

    // // Copy the result back to the host
    // hipMemcpy(output_host, output_dev, output_len * sizeof(unsigned char), hipMemcpyDeviceToHost);

    // // Print the result (in hex format)
    // printf("cSHAKE256 Output: ");
    // for (int i = 0; i < output_len; i++) {
    //     printf("%02x", output_host[i]);
    // }
    // printf("\n");

    // // Clean up memory
    // hipFree(message_dev);
    // hipFree(custom_str_dev);
    // hipFree(output_dev);
    // free(output_host);
}