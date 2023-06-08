/*
 * pcie_utils.h
 *
 *  Created on: Mar 4, 2021
 *      Author: etri_ai3
 */

#ifndef PCIE_UTILS_H_
#define PCIE_UTILS_H_
extern "C" {
uint64_t getopt_integer(char *optarg);
unsigned int reg_read( char *device, long addr);
int reg_write( char *device, long addr, unsigned int data);
ssize_t read_to_buffer(char *fname, int fd, char *buffer, uint64_t size, uint64_t base);
ssize_t write_from_buffer(char *fname, int fd, char *buffer, uint64_t size, uint64_t base);
ssize_t write_one_from_buffer(char *fname, int fd, char *buffer, uint64_t base);
static int timespec_check(struct timespec *t);
void timespec_sub(struct timespec *t1, struct timespec *t2);
int dma_to_device_from_file(char *devname, uint64_t addr, uint64_t size, uint64_t offset, uint64_t count, char *infname, char *ofname);
int dma_to_device(char *devname, uint64_t addr, uint64_t size, uint64_t offset, uint64_t count, uint64_t *buffer);
int dma_from_device_to_file(char *devname, uint64_t addr, uint64_t size, uint64_t offset, uint64_t count, char *ofname);
int dma_from_device(char *devname, uint64_t addr, uint64_t size, uint64_t offset, uint64_t count, uint64_t *buffer);
int compare_file(char *fname1, char *fname2, uint64_t size);
int check_channel();
}


#endif /* PCIE_UTILS_H_ */
