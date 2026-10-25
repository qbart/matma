//! Derive macros for [matma](https://crates.io/crates/matma)'s std140 / std430
//! layouts. Use them through matma (`matma::Std140`, `matma::Std430`; the `derive`
//! feature is on by default) rather than depending on this crate directly.
//!
//! Zero dependencies: the struct is parsed straight from `proc_macro` tokens,
//! which is enough because the input shape is fixed (a non-generic struct with
//! named fields).

use proc_macro::{Delimiter, TokenStream, TokenTree};

/// Generates `NameStd140`, the std140 layout of a plain-data struct.
///
/// ```ignore
/// use matma::*;
///
/// #[derive(Clone, Copy, Std140)]
/// struct Camera {
///     view: float4x4, // offset 0
///     eye: float3,    // offset 64
///     near: f32,      // offset 76 — tucked into the vec3's last lane, like GLSL
/// }
///
/// let gpu: CameraStd140 = camera.as_std140();   // also `Std140<Camera>`
/// queue.write_buffer(&ubo, 0, gpu.as_bytes());  // offsets match the shader block
/// ```
///
/// Requirements: the struct is `Copy`, has named fields, no generics, and every
/// field type implements `matma::layout::Std140Layout` (scalars, vectors,
/// matrices, arrays of those, and other derived structs). The generated struct
/// has the same visibility and field names; field types are the std140 storage
/// (`float3` stays `float3`, `float3x3` becomes padded columns, `[f32; N]` becomes
/// 16-byte slots). It implements `Padded<Name>` (`wrap`/`into_inner`), `From` both
/// ways, `Default`, `Debug`, and exposes `SIZE`, `ALIGN` and `as_bytes()`. Every
/// offset is checked against the GLSL rules at compile time.
#[proc_macro_derive(Std140)]
pub fn derive_std140(input: TokenStream) -> TokenStream {
    expand(input, &STD140)
}

/// Generates `NameStd430`, the std430 layout of a plain-data struct (storage
/// buffers, push constants). Same shape and requirements as [`Std140`](macro@Std140);
/// arrays and small matrices are not rounded up to 16-byte slots.
#[proc_macro_derive(Std430)]
pub fn derive_std430(input: TokenStream) -> TokenStream {
    expand(input, &STD430)
}

struct Mode {
    /// Human name: "std140".
    name: &'static str,
    /// Suffix of the generated struct and conversion method.
    suffix: &'static str,
    /// Full path of the layout trait.
    layout_trait: &'static str,
    /// Whether the trait has a `Slot` associated type.
    has_slot: bool,
    /// Rounds the struct's max member alignment to the struct alignment.
    struct_align_fn: Option<&'static str>,
    /// `repr(align)` to put on the generated struct, if the rule guarantees one.
    repr_align: Option<usize>,
}

const STD140: Mode = Mode {
    name: "std140",
    suffix: "Std140",
    layout_trait: "::matma::layout::Std140Layout",
    has_slot: true,
    struct_align_fn: Some("::matma::layout::std140_struct_align"),
    // std140 structs are always at least 16-aligned; a 32-aligned one (double3/4
    // members) still has a size that is a multiple of 32, so 16 here is never wrong.
    repr_align: Some(16),
};

const STD430: Mode = Mode {
    name: "std430",
    suffix: "Std430",
    layout_trait: "::matma::layout::Std430Layout",
    has_slot: false,
    struct_align_fn: None,
    repr_align: None,
};

struct Field {
    vis: String,
    name: String,
    ty: String,
}

struct Input {
    vis: String,
    name: String,
    fields: Vec<Field>,
}

fn expand(input: TokenStream, mode: &Mode) -> TokenStream {
    match parse(input, mode) {
        Ok(input) => generate(&input, mode)
            .parse()
            .expect("matma-derive generated invalid Rust"),
        Err(msg) => format!("::core::compile_error!({msg:?});")
            .parse()
            .expect("compile_error! is valid Rust"),
    }
}

// ---- parsing ------------------------------------------------------------------

fn parse(input: TokenStream, mode: &Mode) -> Result<Input, String> {
    let mut tokens = input.into_iter().peekable();
    let mut vis = String::new();

    // Outer attributes and visibility, up to `struct`.
    loop {
        match tokens.next() {
            Some(TokenTree::Punct(p)) if p.as_char() == '#' => {
                tokens.next(); // the `[...]` group
            }
            Some(TokenTree::Ident(id)) if id.to_string() == "struct" => break,
            Some(TokenTree::Ident(id)) if id.to_string() == "enum" => {
                return Err(format!("#[derive({})] only supports structs", mode.suffix));
            }
            Some(TokenTree::Ident(id)) if id.to_string() == "union" => {
                return Err(format!("#[derive({})] only supports structs", mode.suffix));
            }
            Some(tt) => {
                vis.push_str(&tt.to_string());
                vis.push(' ');
            }
            None => return Err("expected a struct".into()),
        }
    }

    let name = match tokens.next() {
        Some(TokenTree::Ident(id)) => id.to_string(),
        _ => return Err("expected the struct name".into()),
    };

    let body = match tokens.next() {
        Some(TokenTree::Group(g)) if g.delimiter() == Delimiter::Brace => g.stream(),
        Some(TokenTree::Punct(p)) if p.as_char() == '<' => {
            return Err(format!(
                "#[derive({})] does not support generic structs: the layout of `{name}` must be a single fixed shape",
                mode.suffix
            ));
        }
        Some(TokenTree::Group(g)) if g.delimiter() == Delimiter::Parenthesis => {
            return Err(format!(
                "#[derive({})] needs named fields; `{name}` is a tuple struct",
                mode.suffix
            ));
        }
        _ => {
            return Err(format!(
                "#[derive({})] needs named fields; `{name}` has none",
                mode.suffix
            ));
        }
    };

    let fields = parse_fields(body, mode)?;
    if fields.is_empty() {
        return Err(format!(
            "`{name}` has no fields; a {} block cannot be empty",
            mode.name
        ));
    }

    Ok(Input { vis, name, fields })
}

fn parse_fields(body: TokenStream, mode: &Mode) -> Result<Vec<Field>, String> {
    // Split on top-level commas.
    let mut chunks: Vec<Vec<TokenTree>> = vec![Vec::new()];
    for tt in body {
        match tt {
            TokenTree::Punct(p) if p.as_char() == ',' => chunks.push(Vec::new()),
            tt => chunks.last_mut().expect("non-empty").push(tt),
        }
    }

    let mut fields = Vec::new();
    for chunk in chunks {
        let mut it = chunk.into_iter().peekable();
        // Field attributes.
        while matches!(it.peek(), Some(TokenTree::Punct(p)) if p.as_char() == '#') {
            it.next();
            it.next();
        }
        if it.peek().is_none() {
            continue; // trailing comma
        }
        // Visibility: `pub` optionally followed by `(...)`.
        let mut vis = String::new();
        if matches!(it.peek(), Some(TokenTree::Ident(id)) if id.to_string() == "pub") {
            vis.push_str("pub");
            it.next();
            if matches!(it.peek(), Some(TokenTree::Group(g)) if g.delimiter() == Delimiter::Parenthesis)
            {
                vis.push_str(&it.next().expect("peeked").to_string());
            }
            vis.push(' ');
        }
        let name = match it.next() {
            Some(TokenTree::Ident(id)) => id.to_string(),
            other => {
                return Err(format!(
                    "#[derive({})]: expected a field name, found `{}`",
                    mode.suffix,
                    other.map(|t| t.to_string()).unwrap_or_default()
                ));
            }
        };
        match it.next() {
            Some(TokenTree::Punct(p)) if p.as_char() == ':' => {}
            _ => {
                return Err(format!(
                    "#[derive({})]: expected `:` after field `{name}`",
                    mode.suffix
                ));
            }
        }
        let ty: TokenStream = it.collect();
        if ty.is_empty() {
            return Err(format!(
                "#[derive({})]: field `{name}` has no type",
                mode.suffix
            ));
        }
        fields.push(Field {
            vis,
            name,
            ty: ty.to_string(),
        });
    }
    Ok(fields)
}

// ---- code generation ----------------------------------------------------------

fn generate(input: &Input, mode: &Mode) -> String {
    use std::fmt::Write;

    let Input { vis, name, fields } = input;
    let layout = mode.layout_trait;
    let gen_name = format!("{name}{}", mode.suffix);
    let helper = format!("__matma_layout_{gen_name}");
    let last = fields.len() - 1;
    let mut out = String::new();

    // -- offsets, computed as associated consts on a hidden helper type ----------
    // (not on the generated struct: `Self::X` is not allowed in its field types).
    let _ = writeln!(
        out,
        "#[doc(hidden)] #[allow(non_camel_case_types, dead_code)] {vis}struct {helper};"
    );
    let _ = writeln!(out, "#[allow(dead_code)] impl {helper} {{");
    for (i, f) in fields.iter().enumerate() {
        let prev_end = if i == 0 {
            "0".to_string()
        } else {
            format!("Self::END{}", i - 1)
        };
        let _ = writeln!(
            out,
            "    const A{i}: usize = <{} as {layout}>::ALIGN;",
            f.ty
        );
        let _ = writeln!(
            out,
            "    const S{i}: usize = ::core::mem::size_of::<<{} as {layout}>::Field>();",
            f.ty
        );
        let _ = writeln!(
            out,
            "    const OFF{i}: usize = ::matma::layout::align_up({prev_end}, Self::A{i});"
        );
        let _ = writeln!(out, "    const PAD{i}: usize = Self::OFF{i} - {prev_end};");
        let _ = writeln!(out, "    const END{i}: usize = Self::OFF{i} + Self::S{i};");
    }
    let mut max_align = "Self::A0".to_string();
    for i in 1..fields.len() {
        max_align = format!("::matma::layout::max_usize({max_align}, Self::A{i})");
    }
    let _ = writeln!(out, "    const MAX_ALIGN: usize = {max_align};");
    match mode.struct_align_fn {
        Some(f) => {
            let _ = writeln!(out, "    const ALIGN: usize = {f}(Self::MAX_ALIGN);");
        }
        None => {
            let _ = writeln!(out, "    const ALIGN: usize = Self::MAX_ALIGN;");
        }
    }
    let _ = writeln!(
        out,
        "    const SIZE: usize = ::matma::layout::align_up(Self::END{last}, Self::ALIGN);"
    );
    let _ = writeln!(out, "    const TAIL: usize = Self::SIZE - Self::END{last};");
    let _ = writeln!(out, "}}");

    // -- the generated struct ---------------------------------------------------
    let repr = match mode.repr_align {
        Some(a) => format!("#[repr(C, align({a}))]"),
        None => "#[repr(C)]".to_string(),
    };
    let _ = writeln!(
        out,
        "#[doc = \"The {} layout of [`{name}`]: same fields at the GLSL offsets, with explicit zero padding. Build it with `{name}::as_{}()` or `.into()`; upload `as_bytes()`.\"]",
        mode.name,
        mode.suffix.to_lowercase()
    );
    let _ = writeln!(out, "{repr}");
    let _ = writeln!(out, "#[derive(Clone, Copy)]");
    let _ = writeln!(out, "{vis}struct {gen_name} {{");
    for (i, f) in fields.iter().enumerate() {
        let _ = writeln!(
            out,
            "    #[allow(dead_code)] _pad{i}: [u8; {helper}::PAD{i}],"
        );
        let _ = writeln!(
            out,
            "    #[doc = \"`{}` in {} storage.\"] {}{}: <{} as {layout}>::Field,",
            f.name, mode.name, f.vis, f.name, f.ty
        );
    }
    let _ = writeln!(out, "    #[allow(dead_code)] _tail: [u8; {helper}::TAIL],");
    let _ = writeln!(out, "}}");

    // -- inherent API ------------------------------------------------------------
    let _ = writeln!(out, "impl {gen_name} {{");
    let _ = writeln!(
        out,
        "    #[doc = \"Size in bytes, as the shader sees it (the struct's stride in an array).\"]"
    );
    let _ = writeln!(out, "    pub const SIZE: usize = {helper}::SIZE;");
    let _ = writeln!(
        out,
        "    #[doc = \"{} base alignment of the struct.\"]",
        mode.name
    );
    let _ = writeln!(out, "    pub const ALIGN: usize = {helper}::ALIGN;");
    let _ = writeln!(
        out,
        "    #[doc = \"The struct as bytes, ready to copy into a GPU buffer.\"]"
    );
    let _ = writeln!(out, "    #[inline]");
    let _ = writeln!(out, "    #[allow(unsafe_code)]");
    let _ = writeln!(out, "    pub fn as_bytes(&self) -> &[u8] {{");
    let _ = writeln!(
        out,
        "        // SAFETY: repr(C) plain data whose padding is explicit zeroed fields (the"
    );
    let _ = writeln!(
        out,
        "        // offset/size assertions below prove no implicit padding exists), so every"
    );
    let _ = writeln!(
        out,
        "        // byte is initialised; the slice borrows `self`."
    );
    let _ = writeln!(
        out,
        "        unsafe {{ ::core::slice::from_raw_parts(self as *const Self as *const u8, ::core::mem::size_of::<Self>()) }}"
    );
    let _ = writeln!(out, "    }}");
    let _ = writeln!(out, "}}");

    // -- Padded<Name> for the generated struct -----------------------------------
    let _ = writeln!(
        out,
        "impl ::matma::layout::Padded<{name}> for {gen_name} {{"
    );
    let _ = writeln!(out, "    #[inline]");
    let _ = writeln!(out, "    fn wrap(value: {name}) -> Self {{");
    let _ = writeln!(out, "        Self {{");
    for (i, f) in fields.iter().enumerate() {
        let _ = writeln!(out, "            _pad{i}: [0; {helper}::PAD{i}],");
        let _ = writeln!(
            out,
            "            {0}: <<{1} as {layout}>::Field as ::matma::layout::Padded<{1}>>::wrap(value.{0}),",
            f.name, f.ty
        );
    }
    let _ = writeln!(out, "            _tail: [0; {helper}::TAIL],");
    let _ = writeln!(out, "        }}");
    let _ = writeln!(out, "    }}");
    let _ = writeln!(out, "    #[inline]");
    let _ = writeln!(out, "    fn zeroed() -> Self {{");
    let _ = writeln!(out, "        Self {{");
    for (i, f) in fields.iter().enumerate() {
        let _ = writeln!(out, "            _pad{i}: [0; {helper}::PAD{i}],");
        let _ = writeln!(
            out,
            "            {0}: <<{1} as {layout}>::Field as ::matma::layout::Padded<{1}>>::zeroed(),",
            f.name, f.ty
        );
    }
    let _ = writeln!(out, "            _tail: [0; {helper}::TAIL],");
    let _ = writeln!(out, "        }}");
    let _ = writeln!(out, "    }}");
    let _ = writeln!(out, "    #[inline]");
    let _ = writeln!(out, "    fn into_inner(self) -> {name} {{");
    let _ = writeln!(out, "        {name} {{");
    for f in fields {
        let _ = writeln!(
            out,
            "            {0}: ::matma::layout::Padded::<{1}>::into_inner(self.{0}),",
            f.name, f.ty
        );
    }
    let _ = writeln!(out, "        }}");
    let _ = writeln!(out, "    }}");
    let _ = writeln!(out, "}}");

    // -- std trait impls ----------------------------------------------------------
    let _ = writeln!(out, "impl ::core::convert::From<{name}> for {gen_name} {{");
    let _ = writeln!(out, "    #[inline]");
    let _ = writeln!(
        out,
        "    fn from(value: {name}) -> Self {{ <Self as ::matma::layout::Padded<{name}>>::wrap(value) }}"
    );
    let _ = writeln!(out, "}}");
    let _ = writeln!(out, "impl ::core::convert::From<{gen_name}> for {name} {{");
    let _ = writeln!(out, "    #[inline]");
    let _ = writeln!(
        out,
        "    fn from(value: {gen_name}) -> Self {{ ::matma::layout::Padded::<{name}>::into_inner(value) }}"
    );
    let _ = writeln!(out, "}}");
    let _ = writeln!(out, "impl ::core::default::Default for {gen_name} {{");
    let _ = writeln!(out, "    #[inline]");
    let _ = writeln!(
        out,
        "    fn default() -> Self {{ <Self as ::matma::layout::Padded<{name}>>::zeroed() }}"
    );
    let _ = writeln!(out, "}}");
    let _ = writeln!(out, "impl ::core::fmt::Debug for {gen_name} {{");
    let _ = writeln!(
        out,
        "    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {{"
    );
    let _ = writeln!(out, "        f.debug_struct({gen_name:?})");
    for f in fields {
        let _ = writeln!(out, "            .field({0:?}, &self.{0})", f.name);
    }
    let _ = writeln!(out, "            .finish()");
    let _ = writeln!(out, "    }}");
    let _ = writeln!(out, "}}");

    // -- the layout trait for the source struct, so it nests and arrays ----------
    let _ = writeln!(
        out,
        "// SAFETY: `{gen_name}` is repr(C) plain data with explicit padding (asserted below)."
    );
    let _ = writeln!(out, "unsafe impl {layout} for {name} {{");
    let _ = writeln!(out, "    const ALIGN: usize = {helper}::ALIGN;");
    let _ = writeln!(out, "    type Padded = {gen_name};");
    if mode.has_slot {
        let _ = writeln!(out, "    type Slot = {gen_name};");
    }
    let _ = writeln!(out, "    type Field = {gen_name};");
    let _ = writeln!(out, "}}");

    // -- conversion method on the source struct ----------------------------------
    let method = format!("as_{}", mode.suffix.to_lowercase());
    let _ = writeln!(out, "impl {name} {{");
    let _ = writeln!(
        out,
        "    #[doc = \"This value in {} layout, ready for `as_bytes()`.\"]",
        mode.name
    );
    let _ = writeln!(out, "    #[inline]");
    let _ = writeln!(
        out,
        "    {vis}fn {method}(&self) -> {gen_name} {{ <{gen_name} as ::matma::layout::Padded<{name}>>::wrap(*self) }}"
    );
    let _ = writeln!(out, "}}");

    // -- compile-time proof that Rust placed everything where GLSL does ----------
    let _ = writeln!(out, "const _: () = {{");
    for (i, f) in fields.iter().enumerate() {
        let _ = writeln!(
            out,
            "    assert!(::core::mem::offset_of!({gen_name}, {}) == {helper}::OFF{i}, \"{} offset of `{name}::{}` differs from Rust's placement\");",
            f.name, mode.name, f.name
        );
    }
    let _ = writeln!(
        out,
        "    assert!(::core::mem::size_of::<{gen_name}>() == {helper}::SIZE, \"{} size of `{name}` differs from Rust's\");",
        mode.name
    );
    let _ = writeln!(out, "}};");

    out
}
