#[cfg(feature = "dhatgarbo")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;
use std::{
    error::Error,
    fmt::{Display, Write},
    fs::{self, File},
    hash::Hasher,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    str::FromStr,
    time::Instant,
};

use clap::Parser;
use glsl_lang_pp::processor::{
    ProcessorState,
    event::Event,
    fs::{FileSystem, ParsedFile, Processor, StdProcessor},
    nodes::{Define, DefineObject, ExtensionBehavior},
};

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Name of the person to greet
    #[arg(short, long)]
    file: PathBuf,
    #[arg(short, long)]
    shader_type: ShaderType,
    #[arg(short, long)]
    varying: PathBuf,
    #[arg(short, long, value_parser = clap::value_parser!(Platform))]
    platform: Platform,
    includes: Vec<PathBuf>,
    #[arg(short, long)]
    output: PathBuf,
}

use memchr::memmem::{self, Finder};
#[derive(Debug, Clone)]
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
impl FromStr for Platform {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let save_me = match s {
            "100_es" => Platform::Essl(100),
            "300_es" => Platform::Essl(300),
            "310_es" => Platform::Essl(310),
            "320_es" => Platform::Essl(320),
            "s_4_0" => Platform::Hlsl(400),
            "s_5_0" => Platform::Hlsl(500),
            "s_6_0" => Platform::Dxil(600),
            "s_6_1" => Platform::Dxil(610),
            "s_6_2" => Platform::Dxil(620),
            "s_6_3" => Platform::Dxil(630),
            "s_6_4" => Platform::Dxil(640),
            "s_6_5" => Platform::Dxil(650),
            "s_6_6" => Platform::Dxil(660),
            "s_6_7" => Platform::Dxil(670),
            "s_6_8" => Platform::Dxil(680),
            "s_6_9" => Platform::Dxil(690),
            "metal" => Platform::Metal(1210),
            "metal10-10" => Platform::Metal(1010),
            "metal11-10" => Platform::Metal(1110),
            "metal12-10" => Platform::Metal(1210),
            "metal20-11" => Platform::Metal(2011),
            "metal21-11" => Platform::Metal(2111),
            "metal22-11" => Platform::Metal(2211),
            "metal23-14" => Platform::Metal(2314),
            "metal24-14" => Platform::Metal(2414),
            "metal30-14" => Platform::Metal(3014),
            "metal31-14" => Platform::Metal(3114),
            "pssl" => Platform::Pssl(0),
            "spirv" => Platform::Spirv(1010),
            "spirv10-10" => Platform::Spirv(1010),
            "spirv13-11" => Platform::Spirv(1311),
            "spirv14-11" => Platform::Spirv(1411),
            "spirv15-12" => Platform::Spirv(1512),
            "spirv16-13" => Platform::Spirv(1613),
            "120" => Platform::Glsl(120),
            "130" => Platform::Glsl(130),
            "140" => Platform::Glsl(140),
            "150" => Platform::Glsl(150),
            "330" => Platform::Glsl(330),
            "400" => Platform::Glsl(400),
            "410" => Platform::Glsl(410),
            "420" => Platform::Glsl(420),
            "430" => Platform::Glsl(430),
            "440" => Platform::Glsl(440),
            "wgsl" => Platform::Wglsl(0),
            _ => return Err("What rhe fuck".into()),
        };
        Ok(save_me)
    }
}
#[derive(PartialEq, clap::ValueEnum, Clone, Debug)]
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
    let varyings = get_varyings(&args.varying, &ProcessorState::default()).unwrap();
    let mut procesor = Processor::new();
    *procesor.system_paths_mut() = args.includes.clone();
    let mut file = File::create(&args.output).unwrap();
    let cuh = cuh(
        &args.file,
        &args,
        &mut procesor,
        //        Platform::Essl(310),
        //        ShaderType::Vertex,
        &mut file,
        varyings,
        // args.includes,
    );
    // fs::write(args.output, &cuh).unwrap();
    //    huh(&aah);
    // println!("wah: {aah:?}");
    //    println!("cuh: {cuh:?}");
}
fn get_varyings(path: &Path, _pstate: &ProcessorState) -> Result<Vec<Varying>, Box<dyn Error>> {
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
    args: &Args,
    preprocesor: &mut Processor<TurboStd>,
    writer: &mut impl std::io::Write,
    //    platform: Platform,
    //    shader_type: ShaderType,
    varyings: Vec<Varying>,
    //    incpaths: Vec<PathBuf>,
) {
    //        let aah = io::stdout()
    let time = Instant::now();
    let mut file = fs::read_to_string(filename).unwrap();
    println!("File read: {}ms", time.elapsed().as_micros());
    let mut mem_buffer = String::new();
    let mut cuh = ProcessorState::builder();
    match args.platform {
        Platform::Essl(version) => {
            cuh = cuh.definition(set_def("BGFX_SHADER_LANGUAGE_GLSL", version));
            cuh = cuh.definition(enable_def("BX_PLATFORM_ANDROID"));
            cuh = cuh.definition(set_def("BGFX_SHADER_LANGUAGE_ESSL", version));
        }
        _ => todo!(),
    };
    let type_def = match args.shader_type {
        ShaderType::Compute => enable_def("BGFX_SHADER_TYPE_COMPUTE"),
        ShaderType::Vertex => enable_def("BGFX_SHADER_TYPE_VERTEX"),
        ShaderType::Fragment => enable_def("BGFX_SHADER_TYPE_FRAGMENT"),
    };
    cuh = cuh.definition(type_def);
    cuh = cuh.definition(set_def("M_PI", 3.1415926535897932384626433832795));
    //    cuh = cuh.definition(definition)
    //    preprocesor.parse(path)
    let time = Instant::now();
    let sus = preprocesor.parse_source(&file, filename);
    println!("parse time: {}ms", time.elapsed().as_micros());
    let time = Instant::now();
    //    let iter = sus.process(cuh.clone().finish()).into_iter().flatten();
    println!("Processor iter init {}", time.elapsed().as_micros());
    let time = Instant::now();
    pp_to_token(sus, &mut mem_buffer, cuh.clone().finish());
    println!("cooy time: {}ms", time.elapsed().as_micros());
    // This is mostly the whole purpose of stage 1
    let mut input_varyings = Vec::new();
    let mut input_hash = None;
    let mut output_varyings = Vec::new();
    let mut output_hash = None;
    for line in mem_buffer
        .lines()
        .map(str::trim)
        .filter(|s| s.starts_with('$'))
    {
        if let Some(line) = line.strip_prefix("$input") {
            input_hash = parse_varyingrefs(line, &mut input_varyings);
            // let iter = line.split(',').map(str::trim).map(str::to_string);
            // input_varyings.extend(iter);
        }
        if let Some(line) = line.strip_prefix("$output") {
            output_hash = parse_varyingrefs(line, &mut output_varyings);
            // let iter = line.split(',').map(str::trim).map(str::to_string);
            // output_varyings.extend(iter);
        }
    }
    // println!("{:?}", &input_varyings);
    // println!("{:?}", &output_varyings);
    // println!("{:?}", &varyings);
    let has_fragcolor = find_undocumented(&mem_buffer, &Finder::new("gl_fragData")).is_some();
    let has_frag = find_undocumented(&mem_buffer, &Finder::new("gl_fragColor")).is_some();
    mem_buffer.clear();
    // match shader_type {
    //     ShaderType::Vertex => {
    if let Platform::Essl(number) = args.platform {
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
        if args.shader_type == ShaderType::Vertex
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

    // _ => todo!(),
    // }
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
    const BGFX_BIN_VER: u8 = 11;
    match args.shader_type {
        ShaderType::Vertex => write_header(b'F', output_hash, writer),
        ShaderType::Fragment => write_header(b'F', input_hash, writer),
        ShaderType::Compute => write_header(b'C', output_hash, writer),
    };
    let cuh = cuh.extension("GL_GOOGLE_include_directive", ExtensionBehavior::Enable);
    let stage2 = preprocesor.parse_source(&mem_buffer, &filename);
    pp_to_token(stage2, &mut file, cuh.finish());
    do_transforms(&mut file, &args.shader_type, &args.platform);
    writer.write(&[0, 0]);
    writer.write(&(file.len() as u32).to_le_bytes());
    writer.write_all(file.as_bytes());
    writer.write(&[0]);
    //    return file;
}

fn write_header<T: std::io::Write>(
    typ: u8,
    input_hash: Option<u32>,
    // output_hash: Option<u32>,
    writer: &mut T,
) {
    writer.write(&[typ, b'S', b'H', 11]);
    writer.write(&input_hash.unwrap_or(0).to_le_bytes());
    //    writer.write(&output_hash.unwrap_or(0).to_le_bytes());
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
fn set_def(name: &str, thing: impl ToString) -> Define {
    Define::object(
        name.into(),
        DefineObject::from_str(&thing.to_string()).unwrap(),
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
fn parse_varyingrefs(str: &str, vec: &mut Vec<String>) -> Option<u32> {
    //    for line in str.lines() {
    // if let Some(strip) = str.strip_prefix("$input ") {
    let iter = str.split(",").map(str::trim).map(str::to_string);
    vec.extend(iter);
    println!("{:?}", &vec);
    let mut hasher = HasherM2A::new(0);
    vec.sort_unstable();
    for sus in vec {
        hasher.write(&sus);
    }
    Some(hasher.finish())
    //        println!("what: {}", hasher.finish());
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
        let file = File::open(path)?;
        let cap = file
            .metadata()
            .map(|e| usize::try_from(e.len()).unwrap_or(0))?;
        let mut buf_file = BufReader::new(file);
        let mut line = String::new();
        let mut buf = String::with_capacity(cap);
        let _len = line.len();
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

    const _OES_TEXTURE_3_D: [&str; 4] = [
        "texture3D",
        "texture3DProj",
        "texture3DLod",
        "texture3DProjLod",
    ];

    const EXT_GPU_SHADER4: [&str; 3] = ["gl_VertexID", "gl_InstanceID", "texture2DLodOffset"];

    // To be use from vertex program require:
    // https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_shader_viewport_layer_array.txt
    // DX11 11_1 feature level
    const ARB_SHADER_VIEWPORT_LAYER_ARRAY: [&str; 2] = ["gl_ViewportIndex", "gl_Layer"];

    const ARB_GPU_SHADER5: [&str; 5] = [
        "bitfieldReverse",
        "floatBitsToInt",
        "floatBitsToUint",
        "intBitsToFloat",
        "uintBitsToFloat",
    ];

    const ARB_SHADING_LANGUAGE_PACKING: [&str; 2] = ["packHalf2x16", "unpackHalf2x16"];

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

    const TEXTURE_ARRAY: [&str; 4] = [
        "sampler2DArray",
        "texture2DArray",
        "texture2DArrayLod",
        "shadow2DArray",
    ];

    const ARB_TEXTURE_MULTISAMPLE: [&str; 3] = ["sampler2DMS", "isampler2DMS", "usampler2DMS"];

    const TEXEL_FETCH: [&str; 2] = ["texelFetch", "texelFetchOffset"];

    const BITS_TO_ENCODERS: [&str; 4] = [
        "floatBitsToUint",
        "floatBitsToInt",
        "intBitsToFloat",
        "uintBitsToFloat",
    ];

    const INTEGER_VECS: [&str; 6] = ["ivec2", "uvec2", "ivec3", "uvec3", "ivec4", "uvec4"];
    const S_UNIFORM_TYPE_NAME: [(&str, &str); 4] = [
        ("int", "int"),
        //		NULL,   NULL,
        ("vec4", "float4"),
        ("mat3", "float3x3"),
        ("mat4", "float4x4"),
    ];
    let mut buf = String::new();
    let _uses_lods = has_idents(&code, &ARB_LOD_IDENTS) || has_idents(&code, &EXT_LOD_IDENTS);
    let _uses_texel_fetch = has_idents(&code, &TEXEL_FETCH);
    let _uses_texturems = has_idents(&code, &ARB_TEXTURE_MULTISAMPLE);
    let _uses_gpu_shader = has_idents(&code, &EXT_GPU_SHADER4);
    let uses_texture_arr = has_idents(&code, &TEXTURE_ARRAY);
    let uses_packing = has_idents(&code, &ARB_SHADING_LANGUAGE_PACKING);
    let _uses_viewport = has_idents(&code, &ARB_SHADER_VIEWPORT_LAYER_ARRAY);
    let uses_int_vecs = has_idents(&code, &INTEGER_VECS);
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
                )
                .unwrap();
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
            if has_idents(&code, &ARB_GPU_SHADER5) {
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
                let _TRANSPOSE_POLYFILL: &str = concat!(
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
                buf.push_str(_TRANSPOSE_POLYFILL);
            }
        }
        _ => todo!(),
    }
    code.insert_str(0, &buf);
}
fn has_idents(str: &str, idents: &[&str]) -> bool {
    for ident in idents {
        if let Some(_pos) = memmem::find(str.as_bytes(), ident.as_bytes()) {
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
fn pp_to_token<F: FileSystem>(pp: ParsedFile<'_, F>, buf: &mut String, pstate: ProcessorState) {
    for sus in pp.process(pstate).into_iter().flatten() {
        match sus {
            Event::Error { error, masked } => {
                if !masked {
                    println!("{}", error);
                }
            }
            Event::EnterFile {
                file_id: _,
                path,
                canonical_path: _,
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
                    buf.push_str(token.text());
                }
            }
            _ => {}
        }
    }
}
fn enable_def(name: &str) -> Define {
    Define::object(name.into(), DefineObject::one(), false)
}
