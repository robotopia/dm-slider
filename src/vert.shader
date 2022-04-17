#version 400

in vec2 position;
in vec2 aTexCoord;

uniform mat4 Model;
uniform mat4 View;
uniform mat4 Projection;

out vec2 TexCoord;

void main()
{
    gl_Position = Projection * View * Model * vec4(position, 0.0, 1.0);
    TexCoord = aTexCoord;
}
