#version 400

out vec4 frag_colour;

in vec2 TexCoord;

uniform sampler2D ourTexture;

void main()
{
    frag_colour = texture( ourTexture, TexCoord );
}
