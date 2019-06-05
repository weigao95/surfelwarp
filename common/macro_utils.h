//
// Created by wei on 3/23/18.
//

#pragma once

#define SURFELWARP_NO_COPY_ASSIGN(TypeName)    \
    TypeName(const TypeName&) = delete;        \
    TypeName& operator=(const TypeName&) = delete

#define SURFELWARP_NO_COPY_ASSIGN_MOVE(TypeName)   \
    TypeName(const TypeName&) = delete;            \
    TypeName& operator=(const TypeName&) = delete; \
    TypeName(TypeName&&) = delete;                 \
    TypeName& operator=(TypeName&&) = delete

#define SURFELWARP_DEFAULT_MOVE(TypeName) \
	TypeName(TypeName&&) noexcept = default;       \
	TypeName& operator=(TypeName&&) noexcept = default

#define SURFELWARP_DEFAULT_CONSTRUCT_DESTRUCT(TypeName) \
    TypeName() = default;                               \
    ~TypeName() = default

