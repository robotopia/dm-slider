#version 400

layout (location = 0) in vec2 position;
layout (location = 1) in vec2 aTexCoord;

uniform vec2 tRange;

out vec2 TexCoord;

void main()
{
    // Shift and scale x axis
    float new_x = (position.x - tRange.x) / (tRange.y - tRange.x);

    // Correction for clip space being [-1:1]
    new_x = 2.0*new_x - 1.0;
    float new_y = 2.0*position.y - 1.0;

    gl_Position = vec4(new_x, new_y, 0.0, 1.0);
    TexCoord = aTexCoord;
}
