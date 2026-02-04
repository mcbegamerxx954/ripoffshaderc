#[cfg(feature = "dhatgarbo")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;
use std::{
    any::type_name,
    error::Error,
    fmt::{Display, Pointer, Write},
    fs::{self, File},
    hash::Hasher,
    io::{self, BufRead, BufReader, Read, Stdout},
    os::unix::fs::OpenOptionsExt,
    path::{Path, PathBuf},
    str::{EscapeDefault, FromStr},
    time::Instant,
};

use clap::Parser;
use glsl_lang_pp::{
    exts::DEFAULT_REGISTRY,
    processor::{
        ProcessorState,
        event::Event,
        fs::{FileSystem, Processor, StdProcessor},
        nodes::{Define, DefineObject, ExtensionBehavior},
    },
};

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Name of the person to greet
    #[arg(short, long)]
    file: PathBuf,
    #[arg(short, long)]
    varying: PathBuf,
    includes: Vec<PathBuf>,
    #[arg(short, long)]
    output: PathBuf,
}

use memchr::memmem::{self, Finder, find as mfind};
enum Platform {
    Glsl(u32),
    Essl(u32),
    Hlsl(u32),
    Dxil(u32),
    Metal(u32),
    Spirv(u32),
    Pssl(u32),
    Wglsl(u32),
}
#[derive(PartialEq)]
enum ShaderType {
    Compute,
    Vertex,
    Fragment,
}
//use glsl_lang::lexer::full::fs::PreprocessorExt;
use murmur2::murmur2a;

fn main() {
    #[cfg(feature = "dhatgarbo")]
    let _profiler = dhat::Profiler::new_heap();
    println!("Hello, world!");
    let args = Args::parse();
    let aah = args.file;
    let varyings = get_varyings(&args.varying, &ProcessorState::default()).unwrap();
    //    ah.push(PathBuf::from(vaah));
    //    let aah = Varying::from_line(&aah);
    let cuh = cuh(
        &Path::new(&aah),
        Platform::Essl(310),
        ShaderType::Vertex,
        varyings,
        args.includes,
    );
    fs::write(args.output, &cuh).unwrap();
    //    huh(&aah);
    // println!("wah: {aah:?}");
    //    println!("cuh: {cuh:?}");
}
fn get_varyings(path: &Path, pstate: &ProcessorState) -> Result<Vec<Varying>, Box<dyn Error>> {
    let mut pp = StdProcessor::new();
    let parsed = pp.parse(path)?;
    let mut str = String::new();
    for huh in parsed.into_iter().flatten() {
        match huh {
            Event::Token { token, masked } => {
                if !masked {
                    str.push_str(token.text());
                }
            }
            _ => {}
        }
    }
    println!("{}", &str);
    Ok(Varying::from_str(&str))
}
fn cuh(
    filename: &Path,
    platform: Platform,
    shader_type: ShaderType,
    varyings: Vec<Varying>,
    incpaths: Vec<PathBuf>,
) -> String {
    //        let aah = io::stdout()
    let time = Instant::now();
    let mut file = fs::read_to_string(filename).unwrap();
    println!("File read: {}ms", time.elapsed().as_micros());
    let mut mem_buffer = String::new();
    let mut pp: Processor<TurboStd> = Processor::new();
    let mut cuh = ProcessorState::builder();
    //        .extension("GL_GOOGLE_include_directive", ExtensionBehavior::Enable);
    println!("{:?}", pp.system_paths());
    std::mem::replace(pp.system_paths_mut(), incpaths);
    println!("{:?}", pp.system_paths());
    match platform {
        Platform::Essl(version) => {
            cuh = cuh.definition(version_def("BGFX_SHADER_LANGUAGE_GLSL", version));
            cuh = cuh.definition(Define::object(
                "BX_PLATFORM_ANDROID".into(),
                DefineObject::one(),
                false,
            ));
            cuh = cuh.definition(version_def("BGFX_SHADER_LANGUAGE_ESSL", version));
        }
        _ => todo!(),
    };
    //    pp.parse(path)
    let time = Instant::now();
    let mut sus = pp.parse_source(&file, filename);
    println!("parse time: {}ms", time.elapsed().as_micros());
    let time = Instant::now();
    let iter = sus.process(cuh.clone().finish()).into_iter().flatten();
    println!("Processor iter init {}", time.elapsed().as_micros());
    let time = Instant::now();
    for ev in iter {
        match ev {
            Event::Error { error, masked } => {
                if !masked {
                    println!("{}", error);
                }
            }
            Event::EnterFile {
                file_id,
                path,
                canonical_path,
            } => {
                println!("{:#?}", path);
            }
            Event::Token { token, masked } => {
                if !masked {
                    mem_buffer.push_str(token.text());
                }
            }
            _ => {}
        }
    }
    println!("cooy time: {}ms", time.elapsed().as_micros());
    // This is mostly the whole purpose of stage 1
    let mut input_varyings = Vec::new();
    let mut output_varyings = Vec::new();
    for line in mem_buffer
        .lines()
        .map(str::trim)
        .filter(|s| s.starts_with('$'))
    {
        if let Some(line) = line.strip_prefix("$input") {
            let iter = line.split(',').map(str::trim).map(str::to_string);
            input_varyings.extend(iter);
        }
        if let Some(line) = line.strip_prefix("$output") {
            let iter = line.split(',').map(str::trim).map(str::to_string);
            output_varyings.extend(iter);
        }
    }
    // println!("{:?}", &input_varyings);
    // println!("{:?}", &output_varyings);
    // println!("{:?}", &varyings);
    let has_fragcolor = find_undocumented(&mem_buffer, &Finder::new("gl_fragData")).is_some();
    let has_frag = find_undocumented(&mem_buffer, &Finder::new("gl_fragColor")).is_some();
    mem_buffer.clear();
    match shader_type {
        ShaderType::Vertex => {
            if let Platform::Essl(mut number) = platform {
                //                let frag = Finder::new("gl_fragData[");
                //                let fragcolor = Finder::new("gl_fragColor");

                // let uncomment_iter = source
                //     .lines()
                //     .into_iter()
                //     .map(str::trim)
                //     .filter(|e| e.starts_with("//"));
                // if uncomment_iter
                //     .clone()
                //     .any(|e| frag.find(e.as_bytes()).is_some()
                if number >= 300 {
                    if has_frag {
                        const COLOR_DEF: &str = concat!(
                            "#define gl_FragColor bgfx_FragColor\n",
                            "out mediump vec4 bgfx_FragColor;\n"
                        );
                        mem_buffer.push_str(COLOR_DEF);
                    } else if has_fragcolor {
                        const FRAG_DEF: &str = concat!(
                            "#define gl_FragData bgfx_FragData\n",
                            "out mediump vec4 bgfx_FragData[gl_MaxDrawBuffers];\n"
                        );
                        mem_buffer.push_str(FRAG_DEF);
                    }
                }
                if shader_type == ShaderType::Vertex
                    && input_varyings
                        .iter()
                        .any(|e| !ALLOWED_VERT_INPUTS.contains(&e.as_str()))
                {
                    todo!();
                }
                for huh in input_varyings
                    .iter()
                    .flat_map(|e| varyings.iter().find(|v| v.name.trim() == e.trim()))
                {
                    let name = &huh.name;

                    if name.starts_with("a_") || name.starts_with("i_") {
                        writeln!(
                            &mut mem_buffer,
                            "attribute {}{}{} {name};",
                            OptFormat(huh.precision.as_deref()),
                            OptFormat(huh.interpolation.as_deref()),
                            huh.type_name,
                        )
                        .unwrap();
                    } else {
                        writeln!(
                            &mut mem_buffer,
                            "{}varying {}{} {name};",
                            OptFormat(huh.interpolation.as_deref()),
                            OptFormat(huh.precision.as_deref()),
                            huh.type_name
                        )
                        .unwrap();
                    }
                }
                for huh in output_varyings
                    .iter()
                    .flat_map(|e| varyings.iter().find(|v| v.name.trim() == e.trim()))
                {
                    writeln!(
                        &mut mem_buffer,
                        "{}varying {} {};",
                        OptFormat(huh.interpolation.as_deref()),
                        huh.type_name,
                        huh.name
                    )
                    .unwrap();
                }
            }
        }
        _ => todo!(),
    }
    println!("huhsize: {} \n{mem_buffer}", mem_buffer.len());
    mem_buffer.reserve(file.len());
    mem_buffer.extend(
        file.split_inclusive('\n')
            .filter(|s| !s.trim_start().starts_with('$')),
    );
    //    mem_buffer.push_str(&file);
    file.clear();
    // println!("{mem_buffer}");
    //    mem_buffer.clear();
    //

    let cuh = cuh.extension("GL_GOOGLE_include_directive", ExtensionBehavior::Enable);
    let mut stage2 = pp.parse_source(&mem_buffer, &filename);
    for sus in stage2.process(cuh.finish()).into_iter().flatten() {
        match sus {
            Event::Error { error, masked } => {
                if !masked {
                    println!("{}", error);
                }
            }
            Event::EnterFile {
                file_id,
                path,
                canonical_path,
            } => {
                // let thing = match File::open(&canonical_path) {
                //     Ok(yay) => yay,
                //     Err(e) => {
                //         println!("wtf:  {:?} {e}", canonical_path);
                //         continue;
                //     }
                // };
                // let Ok(metadata) = thing.metadata() else {
                //     continue;
                // };
                // file.reserve(metadata.len() as usize);
                println!("{:#?}", path);
            }
            Event::Token { token, masked } => {
                if !masked {
                    file.push_str(token.text());
                }
            }
            _ => {}
        }
    }
    do_transforms(&mut file, &shader_type, &platform);
    // println!("finalll:{}", &file);
    //    for hi in stage2.process(cuh) {}
    // for token in
    //     let huh: String = pp
    //         .parse(Path::new(filename))
    //         .unwrap()
    //         .process(cuh.clone().finish()).tokenize(300, false, &DEFAULT_REGISTRY)
    //         .into_iter()
    //         .flatten()
    //         .flat_map(|e| e.into_token())
    //         .inspect(|e| print!("{}", e.text())).filter(|e| e.kind())
    //         .map(|e| e.text_range())
    //         .flat_map(|r| source.get(r.start().offset.into()..r.end().offset.into()))
    //         .collect();
    //     //        .take_while(|e| *e != "{")
    //     //        .collect(); // pp.parse(Path::new("hmm")).unwrap().process(cuh.finish());
    //    println!("{buf}");
    return file;

    // file.insert_str(0, &source);
    let mut str = String::with_capacity(file.capacity());
    // // Sucks ass
    // let huh = pp
    //     .parse_source(&file, Path::new(filename))
    //     .process(cuh.finish())
    //     .into_iter()
    //     .flatten()
    //     .flat_map(|e| e.into_token())
    //     .map(|e| e.text_range())
    //     .flat_map(|r| source.get(r.start().offset.into()..r.end().offset.into()));
    // str.extend(huh);
    //    cuh.finish();
}
fn find_undocumented<'a>(haystack: &'a str, needle: &Finder) -> Option<&'a str> {
    let huh = needle.find(haystack.as_bytes())?;
    let backwards = haystack.get(..huh)?.chars().rev().position(|e| e == '\n')?;
    let str = haystack.get(backwards..huh)?;
    if !str.trim().starts_with("//") {
        Some(str)
    } else {
        None
    }
}
fn version_def(name: &str, version: u32) -> Define {
    Define::object(
        name.into(),
        DefineObject::from_str(&version.to_string()).unwrap(),
        false,
    )
}
#[derive(Debug)]
struct Varying {
    precision: Option<String>,
    interpolation: Option<String>,
    name: String,
    type_name: String,
    init: Option<String>,
    semantics: String,
}
impl Varying {
    fn from_str(string: &str) -> Vec<Self> {
        string
            .lines()
            .into_iter()
            //            .filter(|l| l.starts_with("//"))
            .flat_map(Self::from_line)
            .collect()
    }
    fn from_reader<R: BufRead>(reader: &mut R) -> Vec<Self> {
        let mut buf = String::with_capacity(70);
        let mut varyings = Vec::new();
        while reader.read_line(&mut buf).is_ok() {
            if buf.starts_with("//") {
                continue;
            }
            if buf.is_empty() {
                break;
            }
            if let Some(yay) = Self::from_line(&buf) {
                varyings.push(yay);
            }
            buf.clear();
        }
        varyings
    }
    fn from_line(line: &str) -> Option<Self> {
        let line = line.split_once(";").unwrap_or((line, "")).0;
        let mut precision = None;
        let mut interpolation = None;
        let mut iter = line.split_whitespace();
        let mut type_name = None;
        let mut name = None;
        let mut semantics = None;
        let mut init = None;
        for waah in &mut iter {
            if ["lowp", "mediump", "highp"].contains(&waah) {
                precision = Some(waah.to_owned());
            } else if ["flat", "smooth", "noperspective", "centroid"].contains(&waah) {
                interpolation = Some(waah.to_owned());
            } else if type_name.is_none() {
                type_name = Some(waah.to_owned());
            } else {
                name = Some(waah.to_owned());
                break;
            }
        }
        // println!(
        //     "name: {name:?}, ty: {type_name:?}, precisi: {precision:?}, interpo: {interpolation:?}"
        // );
        let mut separator = iter.next()?;
        // println!("sep: {separator}");
        if separator == ":" {
            semantics = Some(iter.next()?.to_owned());
            separator = iter.next().unwrap_or(separator);
        }
        if separator == "=" {
            init = Some(iter.next()?.to_owned());
        }
        Some(Self {
            precision: precision,
            interpolation: interpolation,
            name: name?,
            type_name: type_name?,
            init: init,
            semantics: semantics?,
        })
    }
}
fn huh(str: &str) -> Option<()> {
    //    for line in str.lines() {
    if let Some(strip) = str.strip_prefix("$input ") {
        let mut vec: Vec<String> = strip
            .split(",")
            .map(str::trim)
            .map(str::to_string)
            .collect();
        println!("{:?}", &vec);
        let mut hasher = HasherM2A::new(0);
        vec.sort_unstable();
        for sus in vec {
            hasher.write(&sus);
        }
        println!("what: {}", hasher.finish());
    }
    //  }
    None
}
struct HasherM2A {
    seed: u32,
    buf: Vec<u8>,
}
impl HasherM2A {
    fn new(seed: u32) -> Self {
        Self {
            seed,
            buf: Vec::new(),
        }
    }
    fn finish(&self) -> u32 {
        murmur2a(&self.buf, self.seed)
    }
    fn write(&mut self, bytes: impl AsRef<[u8]>) {
        self.buf.extend_from_slice(bytes.as_ref());
    }
}
struct OptFormat<T>(Option<T>);
impl<T: Display> Display for OptFormat<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(sus) = &self.0 {
            write!(f, "{sus} ")?;
        }
        Ok(())
    }
}
#[derive(Default, Debug, Clone, Copy)]
pub struct TurboStd;

impl FileSystem for TurboStd {
    type Error = std::io::Error;

    fn canonicalize(&self, path: &Path) -> Result<PathBuf, Self::Error> {
        std::fs::canonicalize(path)
    }

    fn exists(&self, path: &Path) -> bool {
        path.exists()
    }

    fn read(&self, path: &Path) -> Result<std::borrow::Cow<'_, str>, Self::Error> {
        let mut file = File::open(path)?;
        let cap = file
            .metadata()
            .map(|e| usize::try_from(e.len()).unwrap_or(0))?;
        let mut buf_file = BufReader::new(file);
        let mut line = String::new();
        let mut buf = String::with_capacity(cap);
        let mut len = line.len();
        loop {
            line.clear();
            match buf_file.read_line(&mut line) {
                Ok(0) => break,
                Ok(_) => {}
                Err(_e) => break,
            }
            let trim = line.trim();
            if trim.starts_with("//") || trim.is_empty() {
                continue;
            }
            buf.push_str(trim);
            buf.push('\n');
        }
        // println!("{buf}");
        Ok(buf.into())
    }
}
fn do_transforms(code: &mut String, stage: &ShaderType, platform: &Platform) {
    const ARB_LOD_IDENTS: [&str; 13] = [
        "texture2DLod",
        "texture2DArrayLod", // BK - interacts with ARB_texture_array.
        "texture2DProjLod",
        "texture2DGrad",
        "texture2DProjGrad",
        "texture3DLod",
        "texture3DProjLod",
        "texture3DGrad",
        "texture3DProjGrad",
        "textureCubeLod",
        "textureCubeGrad",
        "shadow2DLod",
        "shadow2DProjLod",
    ];
    const EXT_LOD_IDENTS: [&str; 6] = [
        "texture2DLod",
        "texture2DProjLod",
        "textureCubeLod",
        "texture2DGrad",
        "texture2DProjGrad",
        "textureCubeGrad",
    ];
    const SHADOW_SAMPLERS: [&str; 3] = ["shadow2D", "shadow2DProj", "sampler2DShadow"];
    const OES: [&str; 3] = ["dFdx", "dFdy", "fwidth"];

    const _OES_texture_3D: [&str; 4] = [
        "texture3D",
        "texture3DProj",
        "texture3DLod",
        "texture3DProjLod",
    ];

    const EXT_gpu_shader4: [&str; 3] = ["gl_VertexID", "gl_InstanceID", "texture2DLodOffset"];

    // To be use from vertex program require:
    // https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_shader_viewport_layer_array.txt
    // DX11 11_1 feature level
    const ARB_shader_viewport_layer_array: [&str; 2] = ["gl_ViewportIndex", "gl_Layer"];

    const ARB_gpu_shader5: [&str; 5] = [
        "bitfieldReverse",
        "floatBitsToInt",
        "floatBitsToUint",
        "intBitsToFloat",
        "uintBitsToFloat",
    ];

    const ARB_shading_language_packing: [&str; 2] = ["packHalf2x16", "unpackHalf2x16"];

    const SUS: [&str; 11] = [
        "uint",
        "uint2",
        "uint3",
        "uint4",
        "isampler2D",
        "usampler2D",
        "isampler3D",
        "usampler3D",
        "isamplerCube",
        "usamplerCube",
        "textureSize",
    ];

    const textureArray: [&str; 4] = [
        "sampler2DArray",
        "texture2DArray",
        "texture2DArrayLod",
        "shadow2DArray",
    ];

    const ARB_texture_multisample: [&str; 3] = ["sampler2DMS", "isampler2DMS", "usampler2DMS"];

    const texelFetch: [&str; 2] = ["texelFetch", "texelFetchOffset"];

    const bitsToEncoders: [&str; 4] = [
        "floatBitsToUint",
        "floatBitsToInt",
        "intBitsToFloat",
        "uintBitsToFloat",
    ];

    const integerVecs: [&str; 6] = ["ivec2", "uvec2", "ivec3", "uvec3", "ivec4", "uvec4"];
    const s_uniformTypeName: [(&str, &str); 4] = [
        ("int", "int"),
        //		NULL,   NULL,
        ("vec4", "float4"),
        ("mat3", "float3x3"),
        ("mat4", "float4x4"),
    ];
    let mut buf = String::new();
    let uses_lods = has_idents(&code, &ARB_LOD_IDENTS) || has_idents(&code, &EXT_LOD_IDENTS);
    let uses_texel_fetch = has_idents(&code, &texelFetch);
    let uses_texturems = has_idents(&code, &ARB_texture_multisample);
    let uses_gpu_shader = has_idents(&code, &EXT_gpu_shader4);
    let uses_texture_arr = has_idents(&code, &textureArray);
    let uses_packing = has_idents(&code, &ARB_shading_language_packing);
    let uses_viewport = has_idents(&code, &ARB_shader_viewport_layer_array);
    let uses_int_vecs = has_idents(&code, &integerVecs);
    match platform {
        Platform::Essl(essl) => {
            let mut essl = *essl;
            if has_idents(&code, &["image2D"]) && essl < 310 {
                essl = 310;
            }
            if essl < 300 && uses_int_vecs {
                essl = 300;
            }

            let in_out = if *stage == ShaderType::Vertex {
                "in"
            } else {
                "out"
            };
            if essl > 100 {
                write!(
                    &mut buf,
                    concat!(
                        "#version {} es\n",
                        "#define attribute in\n",
                        "#define varying {}\n",
                        "precision highp float;\n",
                        "precision highp int;\n"
                    ),
                    essl, in_out
                );
            }
            if essl < 300 && !has_idents(&code, &SHADOW_SAMPLERS) {
                const SHADOW_TEX_DEF: &str = concat!(
                    "#extension GL_EXT_shadow_samplers : enable\n",
                    "#define shadow2D shadow2DEXT\n",
                    "#define shadow2DProj shadow2DProjEXT\n"
                );
                buf.push_str(SHADOW_TEX_DEF);
            } else {
                const SHADOW_DEF: &str = concat!(
                    "#define shadow2D(_sampler, _coord) texture(_sampler, _coord)\n",
                    "#define shadow2DProj(_sampler, _coord) textureProj(_sampler, _coord)\n"
                );
                buf.push_str(SHADOW_DEF);
            };

            if uses_texture_arr && essl >= 300 {
                buf.push_str("precision highp sampler2DArray;\n");
            }
            if has_idents(&code, &ARB_gpu_shader5) {
                buf.push_str("#extension GL_ARB_gpu_shader5 : enable\n");
            }
            if uses_packing {
                buf.push_str("#extension GL_ARB_shading_language_packing : enable\n");
            }
            if essl < 300 && has_idents(&code, &["gl_FragDepth"]) {
                const FRAG_DEF: &str = concat!(
                    "#extension GL_EXT_frag_depth : enable\n",
                    "#define gl_FragDepth gl_FragDepthEXT\n"
                );
                buf.push_str(FRAG_DEF);
            }
            if uses_texture_arr {
                buf.push_str("#extension GL_EXT_texture_array : enable\n");
            }
            if essl == 100 {
                let TRANSPOSE_POLYFILL: &str = concat!(
                    "mat2 transpose(mat2 _mtx)\n",
                    "mat2 transpose(mat2 _mtx)\n",
                    "{\n",
                    "	vec2 v0 = _mtx[0];\n",
                    "	vec2 v1 = _mtx[1];\n",
                    "\n",
                    "	return mat2(\n",
                    "		  vec2(v0.x, v1.x)\n",
                    "		, vec2(v0.y, v1.y)\n",
                    "		);\n",
                    "}\n",
                    "\n",
                    "mat3 transpose(mat3 _mtx)\n",
                    "{\n",
                    "	vec3 v0 = _mtx[0];\n",
                    "	vec3 v1 = _mtx[1];\n",
                    "	vec3 v2 = _mtx[2];\n",
                    "\n",
                    "	return mat3(\n",
                    "		  vec3(v0.x, v1.x, v2.x)\n",
                    "		, vec3(v0.y, v1.y, v2.y)\n",
                    "		, vec3(v0.z, v1.z, v2.z)\n",
                    "		);\n",
                    "}\n",
                    "\n",
                    "mat4 transpose(mat4 _mtx)\n",
                    "{\n",
                    "	vec4 v0 = _mtx[0];\n",
                    "	vec4 v1 = _mtx[1];\n",
                    "	vec4 v2 = _mtx[2];\n",
                    "	vec4 v3 = _mtx[3];\n",
                    "\n",
                    "	return mat4(\n",
                    "		  vec4(v0.x, v1.x, v2.x, v3.x)\n",
                    "		, vec4(v0.y, v1.y, v2.y, v3.y)\n",
                    "		, vec4(v0.z, v1.z, v2.z, v3.z)\n",
                    "		, vec4(v0.w, v1.w, v2.w, v3.w)\n",
                    "		);\n",
                    "}\n"
                );
            }
        }
        _ => todo!(),
    }
    code.insert_str(0, &buf);
}
fn has_idents(str: &str, idents: &[&str]) -> bool {
    for ident in idents {
        if let Some(pos) = memmem::find(str.as_bytes(), ident.as_bytes()) {
            return true;
        }
    }
    false
}
const ALLOWED_VERT_INPUTS: [&str; 23] = [
    "a_position",
    "a_normal",
    "a_tangent",
    "a_bitangent",
    "a_color0",
    "a_color1",
    "a_color2",
    "a_color3",
    "a_indices",
    "a_weight",
    "a_texcoord0",
    "a_texcoord1",
    "a_texcoord2",
    "a_texcoord3",
    "a_texcoord4",
    "a_texcoord5",
    "a_texcoord6",
    "a_texcoord7",
    "i_data0",
    "i_data1",
    "i_data2",
    "i_data3",
    "i_data4",
];
