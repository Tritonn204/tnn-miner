#!/usr/bin/env python3
"""
gen_tiers.py - Split .spec.cpp into per-SIMD-tier translation units.

Each generated TU contains code for one SIMD level with section attributes
for icache locality. Functions get tier-suffixed names (e.g., stage_3_aes)
and explicit section placement. A dispatcher is generated in the header.

Spec format (//@ directives in C++ comments):

  //@ spec <name>                    - name for generated files
  //@ section <prefix>               - enable section placement (.text.<prefix>_<tier>)
  //@ tier <id> <target_features>    - declare a SIMD tier
  //@ common ... //@ end             - code shared across all generated TUs
  //@ decl ... //@ end               - bare function declarations (no target attr)
  //@ begin <tier_id> ... //@ end    - code block for one tier

The generator transforms tier blocks:
  - __attribute__((target("..."))) function definitions get section attrs
    and tier-suffixed names
  - TNN_TARGET_CLONE becomes TNN_SECTION_CLONE with per-target sections
  - Cross-references within a tier are rewritten to use tier-suffixed names
  - For "default" tier: target attr is dropped (section-only)

Generated output (in <output_dir>):
  <name>.gen.h                       - declarations + dispatcher
  <name>_<tier_id>.gen.cpp           - definitions for one tier

Usage:
  python gen_tiers.py <spec_file> <output_dir>
  python gen_tiers.py --check <spec_file> <output_dir>   # exit 0 if up-to-date
  python gen_tiers.py --dry-run <spec_file> <output_dir>  # print output paths
"""

import sys
import os
import re


# =========================================================================
# Balanced-delimiter parser for C macro arguments
# =========================================================================

def extract_macro_args(text, start):
    """Extract comma-separated macro arguments from text starting after '('.
    Handles nested parens, braces, angle brackets, and quoted strings.
    Returns (list_of_arg_strings, end_position_after_closing_paren)."""
    depth_paren = 1
    depth_brace = 0
    args = []
    current = []
    i = start
    in_string = False
    escape = False

    while i < len(text) and depth_paren > 0:
        c = text[i]

        if escape:
            current.append(c)
            escape = False
            i += 1
            continue

        if c == '\\' and in_string:
            current.append(c)
            escape = True
            i += 1
            continue

        if c == '"' and depth_brace == 0:
            in_string = not in_string
            current.append(c)
            i += 1
            continue

        if in_string:
            current.append(c)
            i += 1
            continue

        if c == '(' and depth_brace == 0:
            depth_paren += 1
            current.append(c)
        elif c == ')' and depth_brace == 0:
            depth_paren -= 1
            if depth_paren == 0:
                args.append(''.join(current).strip())
                i += 1
                break
            current.append(c)
        elif c == '{':
            depth_brace += 1
            current.append(c)
        elif c == '}':
            depth_brace -= 1
            current.append(c)
        elif c == ',' and depth_paren == 1 and depth_brace == 0:
            args.append(''.join(current).strip())
            current = []
        else:
            current.append(c)
        i += 1

    return args, i


# =========================================================================
# Function scanning (for declarations + cross-reference rewriting)
# =========================================================================

def scan_target_clones(code):
    """Find TNN_TARGET_CLONE(...) invocations.
    Returns list of (name, ret_type, func_args, body, [target_strings], match_obj)."""
    results = []
    for m in re.finditer(r'TNN_TARGET_CLONE\s*\(', code):
        args, end = extract_macro_args(code, m.end())
        if len(args) < 5:
            continue
        name = args[0].strip()
        ret_type = args[1].strip()
        ret_type = re.sub(r'\bstatic\s+', '', ret_type)
        func_args = args[2].strip()
        body = args[3].strip()
        targets_raw = args[4:]
        results.append((name, ret_type, func_args, body, targets_raw,
                         m.start(), end))
    return results


def scan_attr_target_fns(code):
    """Find __attribute__((target("..."))) function definitions.
    Returns list of (name, ret_type, func_args, target, match_start, match_end)."""
    results = []
    pattern = re.compile(
        r'(__attribute__\s*\(\(\s*target\s*\(\s*"([^"]+)"\s*\)\s*\)\)\s*\n?)'
        r'([\w][\w\s\*&:]*?)\s+(\w+)\s*(\([^{]*?\))\s*\{',
        re.MULTILINE | re.DOTALL
    )
    for m in pattern.finditer(code):
        attr_text = m.group(1)
        target = m.group(2)
        ret_type = m.group(3).strip()
        ret_type = re.sub(r'\bstatic\s+', '', ret_type)
        name = m.group(4)
        func_args = m.group(5).strip()
        results.append((name, ret_type, func_args, target, m.start(), m.end()))
    return results


def collect_function_names(tier_blocks):
    """Collect all function names defined across tier blocks for cross-ref rewriting."""
    names = set()
    for tid, blocks in tier_blocks.items():
        for blk in blocks:
            for m in re.finditer(r'TNN_TARGET_CLONE\s*\(', blk):
                args, _ = extract_macro_args(blk, m.end())
                if len(args) >= 5:
                    names.add(args[0].strip())
            pattern = re.compile(
                r'__attribute__\s*\(\(\s*target\s*\(\s*"[^"]+"\s*\)\s*\)\)\s*\n?'
                r'[\w][\w\s\*&:]*?\s+(\w+)\s*\([^{]*?\)\s*\{',
                re.MULTILINE | re.DOTALL
            )
            for m in pattern.finditer(blk):
                names.add(m.group(1))
    return names


# =========================================================================
# Code transformation for a single tier
# =========================================================================

def make_section_attr(section_prefix, tid):
    """Build the TNN_SECTION(...) string for a tier."""
    return f'TNN_SECTION("{section_prefix}_{tid}")'


def transform_tier_block(code, tid, feats, section_prefix, func_names):
    """Transform a tier block:
    1. Rename functions with tier suffix
    2. Replace __attribute__((target(...))) with target+section (or section-only for default)
    3. Replace TNN_TARGET_CLONE with TNN_SECTION_CLONE
    4. Rewrite cross-references to use tier-suffixed names
    """
    is_default = (feats == 'default')
    section_macro = make_section_attr(section_prefix, tid)

    # --- Transform __attribute__((target("..."))) function definitions ---
    # Work backwards so indices stay valid
    attr_fns = scan_attr_target_fns(code)
    # Sort by start position descending for safe replacement
    for name, ret_type, func_args, target, mstart, mend in sorted(attr_fns, key=lambda x: x[4], reverse=True):
        # Find the full attribute+signature region to replace
        # The pattern matched: __attribute__((target("..."))) ret_type name(args) {
        # We need to replace up to just before the '{'
        brace_pos = code.rfind('{', mstart, mend)
        prefix_end = brace_pos  # everything before the opening brace

        # Build replacement
        if is_default:
            new_prefix = f'{section_macro} {ret_type} {name}_{tid}{func_args} {{'
        else:
            new_prefix = f'__attribute__((target("{target}"))) {section_macro} {ret_type} {name}_{tid}{func_args} {{'

        code = code[:mstart] + new_prefix + code[mend:]

    # --- Expand TNN_TARGET_CLONE → single function with tier's target + section ---
    # FMV (same name + multiple targets) is incompatible with sections.
    # Use the tier's target instead of sub-clone targets. Dispatch picks the tier.
    clones = scan_target_clones(code)
    for name, ret_type, func_args, body, targets_raw, cstart, cend in sorted(clones, key=lambda x: x[5], reverse=True):
        if is_default:
            fn = f'{section_macro} {ret_type} {name}_{tid}{func_args} {body}'
        else:
            fn = f'__attribute__((target("{feats}"))) {section_macro} {ret_type} {name}_{tid}{func_args} {body}'
        code = code[:cstart] + fn + code[cend:]

    # --- Rewrite cross-references (function calls within this tier) ---
    # e.g., stage_3(sp, w) -> stage_3_<tid>(sp, w)
    # Only rewrite names that are defined in the spec, not arbitrary identifiers
    for fname in func_names:
        # Match function calls: fname( but not fname_<something>(
        # Also match function pointers: &fname, or = fname;
        # Use word boundary to avoid partial matches
        # Don't rename the definition sites (already renamed above)
        pattern = re.compile(r'\b' + re.escape(fname) + r'\b(?!_)')
        code = pattern.sub(f'{fname}_{tid}', code)

    return code


# =========================================================================
# Spec parser
# =========================================================================

def parse_spec(path):
    with open(path, 'r') as f:
        lines = f.readlines()

    spec_name = None
    section_prefix = None
    tiers = {}           # id -> target_features
    tier_order = []       # preserve declaration order
    common_blocks = []    # list of code strings
    decl_lines = []       # bare declarations for header
    tier_blocks = {}      # id -> [code strings]

    i = 0
    while i < len(lines):
        s = lines[i].rstrip()

        # //@ spec <name>
        m = re.match(r'^//@ spec\s+(\S+)', s)
        if m:
            spec_name = m.group(1)
            i += 1
            continue

        # //@ section <prefix>
        m = re.match(r'^//@ section\s+(\S+)', s)
        if m:
            section_prefix = m.group(1)
            i += 1
            continue

        # //@ tier <id> <features>
        m = re.match(r'^//@ tier\s+(\w+)\s+(.+)$', s)
        if m:
            tid, feats = m.group(1), m.group(2).strip()
            tiers[tid] = feats
            tier_order.append(tid)
            i += 1
            continue

        # //@ common ... //@ end
        if s.strip() == '//@ common':
            blk = []
            i += 1
            while i < len(lines) and lines[i].rstrip() != '//@ end':
                blk.append(lines[i])
                i += 1
            common_blocks.append(''.join(blk))
            i += 1  # skip //@ end
            continue

        # //@ decl ... //@ end
        if s.strip() == '//@ decl':
            i += 1
            while i < len(lines) and lines[i].rstrip() != '//@ end':
                line = lines[i].rstrip()
                if line and not line.startswith('//'):
                    decl_lines.append(line)
                i += 1
            i += 1  # skip //@ end
            continue

        # //@ begin <tier_id> ... //@ end
        m = re.match(r'^//@ begin\s+(\w+)', s)
        if m:
            tid = m.group(1)
            blk = []
            i += 1
            while i < len(lines) and lines[i].rstrip() != '//@ end':
                blk.append(lines[i])
                i += 1
            tier_blocks.setdefault(tid, []).append(''.join(blk))
            i += 1  # skip //@ end
            continue

        i += 1

    return spec_name, section_prefix, tiers, tier_order, common_blocks, decl_lines, tier_blocks


# =========================================================================
# Code generation
# =========================================================================

def generate(spec_path, output_dir):
    spec_name, section_prefix, tiers, tier_order, common_blocks, decl_lines, tier_blocks = \
        parse_spec(spec_path)

    if not spec_name:
        print("error: no //@ spec directive found", file=sys.stderr)
        sys.exit(1)

    use_sections = section_prefix is not None

    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.basename(spec_path)
    written = []

    # Collect all function names for cross-ref rewriting
    func_names = collect_function_names(tier_blocks) if use_sections else set()

    # --- Collect declarations per tier ---
    # For section mode: declarations use tier-suffixed names
    # For legacy mode: declarations use original FMV style
    tier_decls = {}  # tid -> [(name_suffixed, ret_type, func_args, target_or_none)]

    for tid in tier_order:
        if tid not in tier_blocks:
            continue
        feats = tiers[tid]
        is_default = (feats == 'default')
        decls = []

        for blk in tier_blocks[tid]:
            # Scan TNN_TARGET_CLONE — expanded to single function with tier target
            for name, ret_type, func_args, body, targets_raw, _, _ in scan_target_clones(blk):
                if use_sections:
                    t = None if is_default else feats
                    decls.append((f'{name}_{tid}', ret_type, func_args, t))
                else:
                    # Legacy mode: emit per-target declarations
                    for tr in targets_raw:
                        tr = tr.strip()
                        qm = re.match(r'^"([^"]+)"$', tr)
                        if qm:
                            decls.append((name, ret_type, func_args, qm.group(1)))

            # Scan __attribute__((target(...))) functions
            for name, ret_type, func_args, target, mstart, _ in scan_attr_target_fns(blk):
                # Skip template definitions (internal to the TU)
                preceding = blk[:mstart]
                last_lines = preceding.rstrip().rsplit('\n', 1)[-1]
                if re.search(r'\btemplate\b', last_lines):
                    continue
                if use_sections:
                    t = None if is_default else target
                    decls.append((f'{name}_{tid}', ret_type, func_args, t))
                else:
                    decls.append((name, ret_type, func_args, target))

        tier_decls[tid] = decls

    # --- Generate header ---
    h_path = os.path.join(output_dir, f'{spec_name}.gen.h')
    h_lines = []
    h_lines.append(f'// Generated from {basename} -- DO NOT EDIT\n')
    h_lines.append('#pragma once\n\n')

    for blk in common_blocks:
        h_lines.append(blk)
        h_lines.append('\n')

    # Emit declarations grouped by tier
    for tid in tier_order:
        if tid not in tier_decls:
            continue
        decls = tier_decls[tid]
        if not decls:
            continue

        feats = tiers[tid]
        h_lines.append(f'// --- tier: {tid} ---\n')
        seen = set()
        for name, ret_type, func_args, target in decls:
            key = (name, target)
            if key in seen:
                continue
            seen.add(key)
            if target:
                h_lines.append(f'__attribute__((target("{target}"))) {ret_type} {name}{func_args};\n')
            else:
                h_lines.append(f'{ret_type} {name}{func_args};\n')
        h_lines.append('\n')

    # --- Generate dispatcher (section mode only) ---
    if use_sections:
        # Collect dispatchable functions: one entry per tier per base name
        from collections import OrderedDict
        dispatch_fns = OrderedDict()  # base_name -> [(tid, suffixed_name, ret_type, func_args)]
        for tid in tier_order:
            if tid not in tier_decls:
                continue
            seen_bases = set()
            for name_s, ret_type, func_args, target in tier_decls[tid]:
                if name_s.endswith(f'_{tid}'):
                    base = name_s[:-(len(tid) + 1)]
                else:
                    base = name_s
                if base not in seen_bases:
                    seen_bases.add(base)
                    dispatch_fns.setdefault(base, []).append((tid, f'{base}_{tid}', ret_type, func_args))

        h_lines.append('// --- dispatch ---\n\n')

        for base_name, entries in dispatch_fns.items():
            if len(entries) < 2:
                continue

            ret_type = entries[0][2]
            func_args = entries[0][3]

            h_lines.append(f'// Dispatcher for {base_name}\n')
            h_lines.append(f'using {base_name}_fn = {ret_type}(*){func_args};\n')
            h_lines.append(f'inline {base_name}_fn resolve_{base_name}() {{\n')

            # Check tiers in reverse order (most specific first), default last
            non_default = [(t, n) for t, n, _, _ in entries if tiers[t] != 'default']
            default_entry = [(t, n) for t, n, _, _ in entries if tiers[t] == 'default']

            for tid, name_s in reversed(non_default):
                feats = tiers[tid]
                feat_list = [f.strip() for f in feats.split(',')]
                checks = ' && '.join(f'__builtin_cpu_supports("{f}")' for f in feat_list)
                h_lines.append(f'  if ({checks}) return {name_s};\n')

            if default_entry:
                h_lines.append(f'  return {default_entry[0][1]};\n')
            else:
                h_lines.append(f'  __builtin_unreachable();\n')

            h_lines.append(f'}}\n\n')

            h_lines.append(f'inline {base_name}_fn get_{base_name}() {{\n')
            h_lines.append(f'  static {base_name}_fn fn = resolve_{base_name}();\n')
            h_lines.append(f'  return fn;\n')
            h_lines.append(f'}}\n\n')

    write_if_changed(h_path, ''.join(h_lines))
    written.append(h_path)

    # --- Generate per-tier .cpp files ---
    for tid in tier_order:
        if tid not in tier_blocks:
            continue

        feats = tiers[tid]
        cpp_path = os.path.join(output_dir, f'{spec_name}_{tid}.gen.cpp')
        cpp = []
        cpp.append(f'// Generated from {basename} -- DO NOT EDIT\n')
        cpp.append(f'// Tier: {tid}  features: {feats}\n\n')

        # Define XELIS_TIER_SECTION so headers can section their helpers
        if use_sections:
            sec_name = f'{section_prefix}_{tid}'
            cpp.append(f'// Section placement for helpers included from headers\n')
            cpp.append(f'#ifdef _WIN32\n')
            cpp.append(f'  #define XELIS_TIER_SECTION __attribute__((section(".text${sec_name}")))\n')
            cpp.append(f'#else\n')
            cpp.append(f'  #define XELIS_TIER_SECTION __attribute__((section(".text.{sec_name}")))\n')
            cpp.append(f'#endif\n\n')

        cpp.append(f'#include "{spec_name}.gen.h"\n\n')

        for blk in tier_blocks[tid]:
            if use_sections:
                transformed = transform_tier_block(blk, tid, feats, section_prefix, func_names)
                cpp.append(transformed)
            else:
                cpp.append(blk)
            cpp.append('\n')

        write_if_changed(cpp_path, ''.join(cpp))
        written.append(cpp_path)

    # --- Clean stale .gen.* files from previous runs ---
    written_set = set(os.path.normpath(p) for p in written)
    prefix = f'{spec_name}_'
    for entry in os.listdir(output_dir):
        if not (entry.startswith(spec_name) and '.gen.' in entry):
            continue
        if entry == f'{spec_name}.gen.h' or entry.startswith(prefix):
            full = os.path.normpath(os.path.join(output_dir, entry))
            if full not in written_set:
                os.remove(full)
                print(f"removed stale: {full}", file=sys.stderr)

    return written


def write_if_changed(path, content):
    """Only write if content differs, to avoid unnecessary rebuilds."""
    if os.path.exists(path):
        with open(path, 'r') as f:
            if f.read() == content:
                return False
    with open(path, 'w') as f:
        f.write(content)
    return True


def check_up_to_date(spec_path, output_dir):
    """Return True if all generated files are newer than the spec."""
    if not os.path.exists(spec_path):
        return False
    spec_mtime = os.path.getmtime(spec_path)

    spec_name, _, tiers, tier_order, _, _, tier_blocks = parse_spec(spec_path)
    if not spec_name:
        return False

    h_path = os.path.join(output_dir, f'{spec_name}.gen.h')
    if not os.path.exists(h_path) or os.path.getmtime(h_path) < spec_mtime:
        return False

    for tid in tier_order:
        if tid not in tier_blocks:
            continue
        cpp_path = os.path.join(output_dir, f'{spec_name}_{tid}.gen.cpp')
        if not os.path.exists(cpp_path) or os.path.getmtime(cpp_path) < spec_mtime:
            return False

    return True


def dry_run(spec_path, output_dir):
    """Print output file paths without writing anything."""
    spec_name, _, tiers, tier_order, _, _, tier_blocks = parse_spec(spec_path)
    if not spec_name:
        print("error: no //@ spec directive found", file=sys.stderr)
        sys.exit(1)

    paths = [os.path.join(output_dir, f'{spec_name}.gen.h')]
    for tid in tier_order:
        if tid in tier_blocks:
            paths.append(os.path.join(output_dir, f'{spec_name}_{tid}.gen.cpp'))
    return paths


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == '--check':
        sys.exit(0 if check_up_to_date(sys.argv[2], sys.argv[3]) else 1)

    if len(sys.argv) == 4 and sys.argv[1] == '--dry-run':
        for f in dry_run(sys.argv[2], sys.argv[3]):
            print(f.replace('\\', '/'))
        sys.exit(0)

    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <spec_file> <output_dir>", file=sys.stderr)
        print(f"       {sys.argv[0]} --check <spec_file> <output_dir>",
              file=sys.stderr)
        print(f"       {sys.argv[0]} --dry-run <spec_file> <output_dir>",
              file=sys.stderr)
        sys.exit(1)

    files = generate(sys.argv[1], sys.argv[2])
    for f in files:
        print(f.replace('\\', '/'))
