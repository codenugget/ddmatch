#pragma once

#include <nlohmann/json.hpp>

#include "core/MyVec.h"

namespace utils {
    template<typename T>
    bool parse_optional(const nlohmann::json& j, T&v, const char* name) {
        bool r = j.contains(name);
        if (r)
            v = j[name];
        return r;
    }

    Vec2i parse_Vec2i(const nlohmann::json& j, const char* name) {
        Vec2i ret{0,0};
        int i = 0;
        for (auto& elem : j[name]) {
            if (i > 2)
            return ret;
            ret.x[i] = elem;
            ++i;
        }
        return ret;
    }
} // namespace