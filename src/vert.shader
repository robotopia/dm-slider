#version 400

layout (location = 0) in vec2 position;
layout (location = 1) in vec2 aTexCoord;

//uniform mat4 Model;
//uniform mat4 View;

out vec2 TexCoord;

void main()
{
    //gl_Position = View * Model * vec4(position, 0.0, 1.0);
    gl_Position = vec4(position, 0.0, 1.0);
    TexCoord = aTexCoord;
}
