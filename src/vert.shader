#version 400

in vec3 position;

//uniform mat4 Model;
//uniform mat4 View;
//uniform mat4 Projection;

void main()
{
  gl_Position = vec4(position, 1.0);
  /* gl_Position = Projection * View * Model * vec4(position, 1.0); */
}
