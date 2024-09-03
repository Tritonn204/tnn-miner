/* XMRig
 * Copyright (c) 2018-2024 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2024 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#define APP_ID        "tnn-miner"
#define APP_NAME      "TNN Miner"
#define APP_DESC      "TNN Miner"
#ifndef TNN_VERSION
#define TNN_VERSION "" // for linter to be quiet
#endif
#define APP_VERSION   TNN_VERSION
#define APP_KIND      "miner"

#ifdef _MSC_VER
#   if (_MSC_VER >= 1930)
#       define MSVC_VERSION 2022
#   elif (_MSC_VER >= 1920 && _MSC_VER < 1930)
#       define MSVC_VERSION 2019
#   elif (_MSC_VER >= 1910 && _MSC_VER < 1920)
#       define MSVC_VERSION 2017
#   elif _MSC_VER == 1900
#       define MSVC_VERSION 2015
#   elif _MSC_VER == 1800
#       define MSVC_VERSION 2013
#   elif _MSC_VER == 1700
#       define MSVC_VERSION 2012
#   elif _MSC_VER == 1600
#       define MSVC_VERSION 2010
#   else
#       define MSVC_VERSION 0
#   endif
#endif

#ifdef TNN_OS_WIN
#    define APP_OS "Windows"
#elif defined TNN_OS_IOS
#    define APP_OS "iOS"
#elif defined TNN_OS_MACOS
#    define APP_OS "macOS"
#elif defined TNN_OS_ANDROID
#    define APP_OS "Android"
#elif defined TNN_OS_LINUX
#    define APP_OS "Linux"
#elif defined TNN_OS_FREEBSD
#    define APP_OS "FreeBSD"
#else
#    define APP_OS "Unknown OS"
#endif

#define STR(X) #X
#define STR2(X) STR(X)

#ifdef TNN_ARM
#   define APP_ARCH "ARMv" STR2(TNN_ARM)
#else
#   if defined(__x86_64__) || defined(__amd64__) || defined(_M_X64) || defined(_M_AMD64)
#       define APP_ARCH "x86-64"
#   else
#       define APP_ARCH "x86"
#   endif
#endif

#ifdef TNN_64_BIT
#   define APP_BITS "64 bit"
#else
#   define APP_BITS "32 bit"
#endif