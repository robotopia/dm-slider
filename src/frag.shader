#version 400

out vec4 frag_colour;

in vec2 TexCoord;

uniform sampler2D ourTexture;

vec4 colormap( float x )
{
    float r = clamp(8.0 / 3.0 * x, 0.0, 1.0);
    float g = clamp(8.0 / 3.0 * x - 1.0, 0.0, 1.0);
    float b = clamp(4.0 * x - 3.0, 0.0, 1.0);
    return vec4(r, g, b, 1.0);
}

void main()
{
    vec4 val = texture( ourTexture, TexCoord );
    frag_colour = colormap( val.x );
}
