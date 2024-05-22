


class Jacobi_t {

    public:
      


}



int main(int argc, char ** argv)
{
  mesh_t mesh;

  // Extract topology and domain dimensions from the command-line arguments
  ParseCommandLineArguments(argc, argv, mesh);

  Jacobi_t Jacobi(mesh);
  Jacobi.Run();

  return STATUS_OK;
}
