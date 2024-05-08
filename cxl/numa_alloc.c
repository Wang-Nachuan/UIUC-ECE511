#include <numa.h>
#include <stdlib.h>
#include <stdio.h>

void* allocate_on_node(size_t size, int node) {
    if (numa_available() < 0) {
        fprintf(stderr, "NUMA is not available\n");
        return NULL;
    }
    void* memory = numa_alloc_onnode(size, node);
    return memory;
}

void free_on_node(void* memory) {
    numa_free(memory, 0);
}

int get_numa_node_count() {
    if (numa_available() < 0) {
        fprintf(stderr, "NUMA is not available\n");
        return -1;
    }
    return numa_num_configured_nodes();
}
