#pragma once

#include <cstddef>
#include <cstdint>

// Interface ONLY. No /dev/tcm, /dev/ai_dma or /dev/udma operations exist yet.
// A future SpacemiT CPU kernel will own a TCMHelper per participating core.
// Persistent slot contents may be reused across many prompt tokens; a slot may
// not be overwritten until its previous asynchronous transfer has completed.
// No silent CPU memcpy fallback is permitted for a failed DMA request.
class TCMHelper {
public:
    enum class WeightFamily : uint8_t { Q, KV, WO };

    struct Configuration {
        int cpu_id = -1;
        int tcm_block = -1;
        size_t capacity_bytes = 0;
    };

    struct Transfer {
        uint64_t opaque_id = 0;
        bool valid = false;
    };

    struct Slot {
        void * data = nullptr;
        size_t capacity_bytes = 0;
    };

    virtual ~TCMHelper() = default;

    TCMHelper(const TCMHelper &) = delete;
    TCMHelper & operator=(const TCMHelper &) = delete;

    // Map a TCM block and configure AI-DMA for this core.
    virtual bool initialize(const Configuration & config) = 0;

    // Reserve two or more slots for each family; does NOT copy weights.
    virtual bool reserve(WeightFamily family, size_t bytes_per_slot, unsigned slots) = 0;

    // DMA only. source must be accessible to the selected DMA engine.
    // Returns a ticket immediately; caller can calculate on the other slot.
    virtual bool submit_async(WeightFamily family, unsigned slot,
                              const void * source, size_t bytes, Transfer & ticket) = 0;

    virtual bool wait(const Transfer & ticket) = 0;
    virtual Slot get_slot(WeightFamily family, unsigned slot) const = 0;
    virtual void shutdown() noexcept = 0;

protected:
    TCMHelper() = default;
};
