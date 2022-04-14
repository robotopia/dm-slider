#version 400

in vec2 position;

uniform mat4 Model;
uniform mat4 View;
uniform mat4 Projection;

void main()
{
  gl_Position = Projection * View * Model * vec4(position, 0.0, 1.0);
}
