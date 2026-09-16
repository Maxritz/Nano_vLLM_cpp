#define CACHE_TEST
#include <cassert>
#include <cstdio>
#include "nanovllm/cache.hpp"

int main() {
    std::printf("start\n"); std::fflush(stdout);
    cache::RadixCache rc;
    std::printf("ctor\n"); std::fflush(stdout);

    {
        std::vector<int> seq = {1, 2, 3};
        auto first = rc.insert(seq);
        std::printf("insert1 done\n"); std::fflush(stdout);
        auto second = rc.insert(seq);
        std::printf("insert2 done\n"); std::fflush(stdout);
        assert(first.size() == second.size());
        for (size_t i = 0; i < first.size(); ++i) {
            assert(first[i].slot == second[i].slot);
            assert(second[i].shared == true);
        }
    }
    std::printf("t1 ok\n"); std::fflush(stdout);
    {
        std::vector<int> seq = {1, 2, 3};
        auto inserted = rc.insert(seq);
        auto looked = rc.lookup(seq);
        assert(looked.size() == inserted.size());
        for (size_t i = 0; i < inserted.size(); ++i)
            assert(looked[i].slot == inserted[i].slot);
    }
    std::printf("t2 ok\n"); std::fflush(stdout);
    {
        assert(rc.size() > 0);
        rc.clear();
        assert(rc.size() == 0);
    }
    std::printf("All cache tests passed.\n");
    return 0;
}