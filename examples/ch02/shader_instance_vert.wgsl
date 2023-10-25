// vertex shader
@group(0) @binding(0)  var<uniform> vpMat: mat4x4f;
@group(0) @binding(1)  var<storage> modelMat: array<mat4x4f>;
@group(0) @binding(2)  var<storage> normalMat: array<mat4x4f>;

struct Input {
    @builtin(instance_index) idx: u32, 
    @location(0) position: vec4f, 
    @location(1) normal: vec4f,
    @location(2) color: vec4f,
};

struct Output {
    @builtin(position) position: vec4f,
    @location(0) vPosition: vec4f,
    @location(1) vNormal: vec4f,
    @location(2) vColor: vec4f,
};

@vertex
fn vs_main(in: Input) -> Output {    
    var output: Output;     
    let modelMat = modelMat[in.idx];
    let normalMat = normalMat[in.idx];
    let mPosition = modelMat * in.position; 
    output.vPosition = mPosition;                  
    output.vNormal =  normalMat * in.normal;
    output.position = vpMat * mPosition;   
    output.vColor = in.color;            
    return output;
}