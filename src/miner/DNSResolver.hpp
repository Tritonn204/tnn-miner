/*
    This file is part of CPPDNSResolver.
    Copyright (C) 2020 ReimuNotMoe

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of the
    License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#pragma once

#include <stdexcept>
#include <system_error>
#include <initializer_list>
#include <string_view>

#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <unistd.h>

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>

#include <udns.h>

#include <boost/asio.hpp>
#include <boost/bind/bind.hpp>

#define THROW_ERRNO	throw std::system_error(errno, std::system_category(), strerror(errno))

namespace YukiWorkshop {
	using namespace boost::placeholders;
	class DNSResolver {
	public:
		struct MXRecord {
			int priority;
			std::string_view name;

			MXRecord(int __prio, char *__name) : priority(__prio), name((const char *)__name) {

			}
		};

		struct SRVRecord {
			int priority;
			int weight;
			int port;
			std::string_view name;

			SRVRecord(int __prio, int __weight, int __port, char *__name) : priority(__prio), weight(__weight), port(__port), name((const char *)__name) {

			}
		};

		// Error, Results, Query Name, CNAME, TTL
		typedef std::function<void(int, std::vector<boost::asio::ip::address_v4> &, const std::string_view &, const std::string_view &, uint)> A4Callback;
		typedef std::function<void(int, std::vector<boost::asio::ip::address_v6> &, const std::string_view &, const std::string_view &, uint)> A6Callback;
		typedef std::function<void(int, const std::vector<std::string_view> &, const std::string_view &, const std::string_view &, uint)> PtrCallback;
		typedef std::function<void(int, const std::vector<std::string_view> &, const std::string_view &, const std::string_view &, uint)> TXTCallback;
		typedef std::function<void(int, const std::vector<MXRecord> &, const std::string_view &, const std::string_view &, uint)> MXCallback;
		typedef std::function<void(int, const std::vector<SRVRecord> &, const std::string_view &, const std::string_view &, uint)> SRVCallback;

	private:
		dns_ctx *ctx_udns = nullptr;
		int fd_udns = -1;

		boost::asio::io_service &asio_iosvc;
		std::unique_ptr<boost::asio::ip::udp::socket> asio_socket;

		uint32_t requests_pending = 0;

		void __init();
		void __fini();
		void __open();

		void io_wait_read();
		void iocb_read_avail();

		void set_servers(const std::initializer_list<std::string> &__nameservers);
		void post_resolve();

		static void dnscb_a4(struct dns_ctx *ctx, struct dns_rr_a4 *result, void *data);
		static void dnscb_a6(struct dns_ctx *ctx, struct dns_rr_a6 *result, void *data);
		static void dnscb_txt(struct dns_ctx *ctx, struct dns_rr_txt *result, void *data);
		static void dnscb_mx(struct dns_ctx *ctx, struct dns_rr_mx *result, void *data);
		static void dnscb_srv(struct dns_ctx *ctx, struct dns_rr_srv *result, void *data);
		static void dnscb_ptr(struct dns_ctx *ctx, struct dns_rr_ptr *result, void *data);

	public:
		explicit DNSResolver(boost::asio::io_service &__io_svc);

		DNSResolver(boost::asio::io_service &__io_svc, const std::initializer_list<std::string> &__nameservers);

		~DNSResolver();

		void resolve_a4(const std::string &__hostname, const A4Callback &__callback); // A
		void resolve_a6(const std::string &__hostname, const A6Callback &__callback); // AAAA
		void resolve_a4ptr(const boost::asio::ip::address_v4 &__addr, const PtrCallback &__callback);
		void resolve_a6ptr(const boost::asio::ip::address_v6 &__addr, const PtrCallback &__callback);
		void resolve_txt(const std::string &__hostname, const TXTCallback &__callback);
		void resolve_mx(const std::string &__hostname, const MXCallback &__callback);
		void resolve_srv(const std::string &__hostname, const std::string &__srv, const std::string &__proto, const SRVCallback &__callback);

		static const std::string& error_string(int __err);

	};
}
