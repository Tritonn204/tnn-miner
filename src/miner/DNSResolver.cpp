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

#ifndef _WIN32

#include "DNSResolver.hpp"


using namespace YukiWorkshop;

void DNSResolver::__init() {
	dns_init(nullptr, 0);

	ctx_udns = dns_new(nullptr);
	if (!ctx_udns)
		throw std::system_error(ENOMEM, std::system_category(), strerror(ENOMEM));

	if (dns_init(ctx_udns, 0) < 0)
		THROW_ERRNO;
}

void DNSResolver::__fini() {
	if (ctx_udns) {
		// fprintf(stderr,"__fini\n");
		dns_free(ctx_udns);
		ctx_udns = nullptr;
	}
}

void DNSResolver::__open() {
	if ((fd_udns = dns_open(ctx_udns)) < 0) {
		THROW_ERRNO;
	}

	struct sockaddr sa;
	socklen_t len = sizeof(sa);
	if (getsockname(fd_udns, &sa, &len))
		throw std::system_error(std::error_code(errno, std::system_category()), "getsockname");

	asio_socket = std::make_unique<boost::asio::ip::udp::socket>(asio_iosvc,
								     sa.sa_family == AF_INET ?
								     boost::asio::ip::udp::v4()
											     : boost::asio::ip::udp::v6(),
								     dns_sock(ctx_udns));
}

void DNSResolver::io_wait_read() {
	// fprintf(stderr,"io_wait_read\n");
	asio_socket->async_receive(boost::asio::null_buffers(),
				   boost::bind(&DNSResolver::iocb_read_avail, this));
}

void DNSResolver::iocb_read_avail() {
	// fprintf(stderr,"iocb_read_avail\n");
	dns_ioevent(ctx_udns, time(nullptr));

	if (requests_pending)
		io_wait_read();
}

void DNSResolver::set_servers(const std::initializer_list<std::string> &__nameservers) {
	dns_add_serv(ctx_udns, nullptr);

	for (auto &it : __nameservers) {
		if (dns_add_serv(ctx_udns, it.c_str()) < 0) {
			THROW_ERRNO;
		}
	}
}

void DNSResolver::post_resolve() {
	requests_pending++;
	dns_timeouts(ctx_udns, -1, time(nullptr));
	io_wait_read();
}

void DNSResolver::dnscb_a4(struct dns_ctx *ctx, struct dns_rr_a4 *result, void *data) {
	auto *pd = (std::pair<DNSResolver *, A4Callback> *) data;
	pd->first->requests_pending--;

	std::vector<boost::asio::ip::address_v4> addrs;

	if (result) {
		for (uint32_t i = 0; i < result->dnsa4_nrr; i++) {
			std::array<unsigned char, 4> buf;
			memcpy(buf.data(), &result->dnsa4_addr[i].s_addr, 4);
			addrs.emplace_back(buf);
		}

		std::string_view cname(result->dnsa4_cname);
		std::string_view qname(result->dnsa4_qname);

		pd->second(DNS_E_NOERROR, addrs, qname, cname, result->dnsa4_ttl);
		free(result);
	} else {
		pd->second(dns_status(pd->first->ctx_udns), addrs, {}, {}, 0);
	}

	delete pd;
}

void DNSResolver::dnscb_a6(struct dns_ctx *ctx, struct dns_rr_a6 *result, void *data) {
	auto *pd = (std::pair<DNSResolver *, A6Callback> *) data;
	pd->first->requests_pending--;

	std::vector<boost::asio::ip::address_v6> addrs;

	if (result) {
		for (uint32_t i = 0; i < result->dnsa6_nrr; i++) {
			std::array<unsigned char, 16> buf;
			memcpy(buf.data(), &result->dnsa6_addr[i], 16);
			addrs.emplace_back(buf);
		}

		std::string_view cname(result->dnsa6_cname);
		std::string_view qname(result->dnsa6_qname);

		pd->second(DNS_E_NOERROR, addrs, qname, cname, result->dnsa6_ttl);
		free(result);
	} else {
		pd->second(dns_status(pd->first->ctx_udns), addrs, {}, {}, 0);
	}

	delete pd;
}

void DNSResolver::dnscb_txt(struct dns_ctx *ctx, struct dns_rr_txt *result, void *data) {
	auto *pd = (std::pair<DNSResolver *, TXTCallback> *) data;
	pd->first->requests_pending--;

	if (result) {
		std::vector<std::string_view> addrs;

		for (uint32_t i = 0; i < result->dnstxt_nrr; i++) {
			addrs.emplace_back((const char *) result->dnstxt_txt[i].txt, result->dnstxt_txt[i].len);
		}

		std::string_view cname(result->dnstxt_cname);
		std::string_view qname(result->dnstxt_qname);

		pd->second(DNS_E_NOERROR, addrs, qname, cname, result->dnstxt_ttl);
		free(result);
	} else {
		pd->second(dns_status(pd->first->ctx_udns), {}, {}, {}, 0);
	}

	delete pd;
}

void DNSResolver::dnscb_mx(struct dns_ctx *ctx, struct dns_rr_mx *result, void *data) {
	auto *pd = (std::pair<DNSResolver *, MXCallback> *) data;
	pd->first->requests_pending--;

	if (result) {
		std::vector<MXRecord> addrs;

		for (uint32_t i = 0; i < result->dnsmx_nrr; i++) {
			addrs.emplace_back(result->dnsmx_mx[i].priority, result->dnsmx_mx[i].name);
		}

		std::string_view cname(result->dnsmx_cname);
		std::string_view qname(result->dnsmx_qname);

		pd->second(DNS_E_NOERROR, addrs, qname, cname, result->dnsmx_ttl);
		free(result);
	} else {
		pd->second(dns_status(pd->first->ctx_udns), {}, {}, {}, 0);
	}

	delete pd;
}

void DNSResolver::dnscb_srv(struct dns_ctx *ctx, struct dns_rr_srv *result, void *data) {
	auto *pd = (std::pair<DNSResolver *, SRVCallback> *) data;
	pd->first->requests_pending--;

	if (result) {
		std::vector<SRVRecord> addrs;

		for (uint32_t i = 0; i < result->dnssrv_nrr; i++) {
			auto &r = result->dnssrv_srv[i];
			addrs.emplace_back(r.priority, r.weight, r.port, r.name);
		}

		std::string_view cname(result->dnssrv_cname);
		std::string_view qname(result->dnssrv_qname);

		pd->second(DNS_E_NOERROR, addrs, qname, cname, result->dnssrv_ttl);
		free(result);
	} else {
		pd->second(dns_status(pd->first->ctx_udns), {}, {}, {}, 0);
	}

	delete pd;
}

void DNSResolver::dnscb_ptr(struct dns_ctx *ctx, struct dns_rr_ptr *result, void *data) {
	auto *pd = (std::pair<DNSResolver *, PtrCallback> *) data;
	pd->first->requests_pending--;

	if (result) {
		std::vector<std::string_view> addrs;

		for (uint32_t i = 0; i < result->dnsptr_nrr; i++) {
			addrs.emplace_back(result->dnsptr_ptr[i]);
		}

		std::string_view cname(result->dnsptr_cname);
		std::string_view qname(result->dnsptr_qname);

		pd->second(DNS_E_NOERROR, addrs, qname, cname, result->dnsptr_ttl);
		free(result);
	} else {
		pd->second(dns_status(pd->first->ctx_udns), {}, {}, {}, 0);
	}

	delete pd;
}

DNSResolver::DNSResolver(boost::asio::io_service &__io_svc) : asio_iosvc(__io_svc) {
	__init();
	__open();
}

DNSResolver::DNSResolver(boost::asio::io_service &__io_svc, const std::initializer_list<std::string> &__nameservers)
	: asio_iosvc(__io_svc) {
	__init();
	set_servers(__nameservers);
	__open();
}

DNSResolver::~DNSResolver() {
	__fini();
}

void DNSResolver::resolve_a4(const std::string &__hostname, const DNSResolver::A4Callback &__callback) {
	auto *pd = new std::pair<DNSResolver *, A4Callback>(this, __callback);

	fprintf(stderr,"resolve_a4: %s\n", __hostname.c_str());

	dns_submit_a4(ctx_udns, __hostname.c_str(), 0, &DNSResolver::dnscb_a4, pd);
	post_resolve();
}

void DNSResolver::resolve_a6(const std::string &__hostname, const DNSResolver::A6Callback &__callback) {
	auto *pd = new std::pair<DNSResolver *, A6Callback>(this, __callback);

	fprintf(stderr,"resolve_a6: %s\n", __hostname.c_str());

	dns_submit_a6(ctx_udns, __hostname.c_str(), 0, &DNSResolver::dnscb_a6, pd);
	post_resolve();
}

void DNSResolver::resolve_a4ptr(const boost::asio::ip::address_v4 &__addr, const DNSResolver::PtrCallback &__callback) {
	auto *pd = new std::pair<DNSResolver *, PtrCallback>(this, __callback);

	fprintf(stderr,"resolve_a4ptr: %s\n", __addr.to_string().c_str());

	dns_submit_a4ptr(ctx_udns, (const struct in_addr *)__addr.to_bytes().data(), &DNSResolver::dnscb_ptr, pd);
	post_resolve();
}

void DNSResolver::resolve_a6ptr(const boost::asio::ip::address_v6 &__addr, const DNSResolver::PtrCallback &__callback) {
	auto *pd = new std::pair<DNSResolver *, PtrCallback>(this, __callback);

	fprintf(stderr,"resolve_a6ptr: %s\n", __addr.to_string().c_str());

	dns_submit_a6ptr(ctx_udns, (const struct in6_addr *)__addr.to_bytes().data(), &DNSResolver::dnscb_ptr, pd);
	post_resolve();
}

void DNSResolver::resolve_txt(const std::string &__hostname, const DNSResolver::TXTCallback &__callback) {
	auto *pd = new std::pair<DNSResolver *, TXTCallback>(this, __callback);

	fprintf(stderr,"resolve_txt: %s\n", __hostname.c_str());

	dns_submit_txt(ctx_udns, __hostname.c_str(), 0, 0, &DNSResolver::dnscb_txt, pd);
	post_resolve();
}

void DNSResolver::resolve_mx(const std::string &__hostname, const DNSResolver::MXCallback &__callback) {
	auto *pd = new std::pair<DNSResolver *, MXCallback>(this, __callback);

	fprintf(stderr,"resolve_mx: %s\n", __hostname.c_str());

	dns_submit_mx(ctx_udns, __hostname.c_str(), 0, &DNSResolver::dnscb_mx, pd);
	post_resolve();
}

void DNSResolver::resolve_srv(const std::string &__hostname, const std::string &__srv, const std::string &__proto,
			      const DNSResolver::SRVCallback &__callback) {
	auto *pd = new std::pair<DNSResolver *, SRVCallback>(this, __callback);

	fprintf(stderr, "resolve_srv: %s\n", __hostname.c_str());

	dns_submit_srv(ctx_udns, __hostname.c_str(), __srv.c_str(), __proto.c_str(), 0, &DNSResolver::dnscb_srv, pd);
	post_resolve();
}

static const std::string errstr_none = "No error";
static const std::string errstr_tempfail = "Server timeout or down";
static const std::string errstr_protocol = "Malformed reply";
static const std::string errstr_nodata = "Domain exists but no data of requested type";
static const std::string errstr_nomem = "Out of memory";
static const std::string errstr_badquery = "Malformed query";
static const std::string errstr_nxdomain = "Domain does not exist";
static const std::string errstr_invalid = "Invalid error";

const std::string &DNSResolver::error_string(int __err) {
	switch (__err) {
		case DNS_E_NOERROR:
			return errstr_none;
		case DNS_E_TEMPFAIL:
			return errstr_tempfail;
		case DNS_E_PROTOCOL:
			return errstr_protocol;
		case DNS_E_NXDOMAIN:
			return errstr_nxdomain;
		case DNS_E_NODATA:
			return errstr_nodata;
		case DNS_E_NOMEM:
			return errstr_nomem;
		case DNS_E_BADQUERY:
			return errstr_badquery;
		default:
			return errstr_invalid;
	}
}

#endif