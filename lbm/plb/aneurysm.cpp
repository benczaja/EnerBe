/* This file is part of the Palabos library.
 *
 * The Palabos softare is developed since 2011 by FlowKit-Numeca Group Sarl
 * (Switzerland) and the University of Geneva (Switzerland), which jointly
 * own the IP rights for most of the code base. Since October 2019, the
 * Palabos project is maintained by the University of Geneva and accepts
 * source code contributions from the community.
 * 
 * Contact:
 * Jonas Latt
 * Computer Science Department
 * University of Geneva
 * 7 Route de Drize
 * 1227 Carouge, Switzerland
 * jonas.latt@unige.ch
 *
 * The most recent release of Palabos can be downloaded at 
 * <https://palabos.unige.ch/>
 *
 * The library Palabos is free software: you can redistribute it and/or
 * modify it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * The library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

/** \file
  * This code solves the steady flow inside an aneurysm. It introduces several
  * new concepts like Guo off lattice boundary conditions, reading of
  * surface geometry STL files, smooth grid refinement and voxelization.
  * Make sure to unpack the file aneurysm.stl.tgz before running the
  * simulation.
  **/

#include "palabos3D.h"
#include "palabos3D.hh"
#include <pmt.h> // needed for PMT

using namespace plb;
using namespace std;

typedef double T;
typedef Array<T,3> Velocity;
#define DESCRIPTOR descriptors::D3Q19Descriptor

plint extraLayer      = 0;  // Make the bounding box larger; for visualization purposes
                            //   only. For the simulation, it is OK to have extraLayer=0.
const plint blockSize = 20; // Zero means: no sparse representation.
const plint envelopeWidth = 1;  // For standard BGK dynamics.
const plint extendedEnvelopeWidth = 2;  // Because the Guo off lattice boundary condition
                                        //   needs 2-cell neighbor access.

bool performOutput = false;
bool doImages = false;
bool useAllDirections = false;
bool useRegularizedWall = false;
bool useIncompressible = false;
bool poiseuilleInlet = false;
bool convectiveScaling = false;

T kinematicViscosity       = 0.;
T averageInletVelocity     = 0.;
plint referenceResolution  = 0;
T nuLB                     = 0.;
T fluidDensity             = 0.;
T volume                   = 0.;
T userDefinedInletDiameter = 0.;

plint referenceDirection = 0;
plint openingSortDirection = 0;

plint numIter = 0;
plint startLevel = 0;
T epsilon = 0.;

TriangleSet<T>* triangleSet = 0;
T currentTime = 0;

// Structure which defines an ``opening''. The surface geometry of the aneurysm,
//   as given by the user in the form of an STL file, contains holes, which in 
//   the specific simulation represent inlets and outlets.
template<typename T>
struct Opening {
    bool inlet;
    Array<T,3> center;
    T innerRadius;
};

std::vector<Opening<T> > openings;

void iniLattice( MultiBlockLattice3D<T,DESCRIPTOR>& lattice,
                 VoxelizedDomain3D<T>& voxelizedDomain )
{
    // Switch all remaining outer cells to no-dynamics, except the outer
    //   boundary layer, and keep the rest as BGKdynamics.
    defineDynamics(lattice, voxelizedDomain.getVoxelMatrix(), lattice.getBoundingBox(),
                   new NoDynamics<T,DESCRIPTOR>, voxelFlag::outside);
    initializeAtEquilibrium(lattice, lattice.getBoundingBox(), (T) 1., Array<T,3>((T) 0.,(T) 0.,(T) 0.));
    lattice.initialize();
}

// This function assigns proper boundary conditions to the openings of the surface geometry
//   of the aneurysm. Which opening is inlet and which is outlet is defined by the user in
//   the input XML file. For the inlet, there is a choice between a Poiseuille velocity
//   profile and a simple plug velocity profile. At the outlets a Neumann boundary
//   condition with constant pressure is prescribed.
void setOpenings (
    std::vector<BoundaryProfile3D<T,Velocity>*>& inletOutlets,
    TriangleBoundary3D<T>& boundary, T uLB)
{
    for (pluint i=0; i<openings.size(); ++i) {
        Opening<T>& opening = openings[i];
        opening.center = computeBaryCenter (
                boundary.getMesh(),
                boundary.getInletOutlet(openingSortDirection)[i] );
        opening.innerRadius = computeInnerRadius (
                boundary.getMesh(),
                boundary.getInletOutlet(openingSortDirection)[i] );

        if (opening.inlet) {
            if (poiseuilleInlet) {
                inletOutlets.push_back (
                        new PoiseuilleProfile3D<T>(uLB) );
            }
            else {
                inletOutlets.push_back (
                        new VelocityPlugProfile3D<T>(uLB) );
            }
        }
        else {
            inletOutlets.push_back (
                    new DensityNeumannBoundaryProfile3D<T> );
        }
    }
}

// This function outputs velocity, vorticity and pressure data, at selected
//   points of the computational domain, given their coordinates in physical units.
std::vector<T> pointMeasures (
        MultiBlockLattice3D<T,DESCRIPTOR>& lattice,
        Array<T,3> location, T dx, T dt )
{
    std::vector<Array<T,3> > physicalPositions, positions;
    physicalPositions.push_back(Array<T,3>(0.022046, 0.015072, 0.044152));
    physicalPositions.push_back(Array<T,3>(0.027132, 0.049947, 0.095012));
    physicalPositions.push_back(Array<T,3>(0.034398, 0.056487, 0.057957));
    physicalPositions.push_back(Array<T,3>(0.031492, 0.025971, 0.084113));
    physicalPositions.push_back(Array<T,3>(0.025679, 0.025971, 0.091379));
    physicalPositions.push_back(Array<T,3>(0.018413, 0.011439, 0.076848));
    positions.resize(physicalPositions.size());

    for (pluint i=0; i<physicalPositions.size(); ++i) {
        positions[i] = (physicalPositions[i]-location)/dx;
    }

    std::vector<Array<T,3> > velocities = velocitySingleProbes(lattice, positions);
    std::vector<Array<T,3> > vorticities = vorticitySingleProbes(lattice, positions);
    std::vector<T> densities = densitySingleProbes(lattice, positions);

    std::vector<T> data;
    for (pluint i=0; i<physicalPositions.size(); ++i) {
        Array<T,3> pos = physicalPositions[i];
        Array<T,3> vel = velocities[i]*dx/dt;
        Array<T,3> vort = vorticities[i]/dt;
        T pressure = DESCRIPTOR<T>::cs2*(densities[i]-1.)*dx*dx/(dt*dt)*fluidDensity;
        if (performOutput) {
            pcout << "Pos ("
                  << pos[0] << "," << pos[1] << "," << pos[2]
                  << "); Velocity ("
                  << vel[0] << "," << vel[1] << "," << vel[2]
                  << "); Vorticity ("
                  << vort[0] << "," << vort[1] << "," << vort[2]
                  << "); Pressure " << pressure << std::endl;
        }
        data.push_back(norm(vel));
        data.push_back(norm(vort));
        data.push_back(pressure);
    }
    return data;
}

void writeImages (
         OffLatticeBoundaryCondition3D<T,DESCRIPTOR,Velocity>& boundaryCondition,
         Box3D const& imageDomain, Box3D const& vtkDomain, std::string fname, Array<T,3> location, T dx, T dt )
{
    VtkImageOutput3D<T> vtkOut(fname, dx, location);
    vtkOut.writeData<float>(*boundaryCondition.computePressure(vtkDomain), "p", util::sqr(dx/dt)*fluidDensity);
    vtkOut.writeData<float>(*boundaryCondition.computeVelocityNorm(vtkDomain), "u", dx/dt);
    vtkOut.writeData<float>(*copyConvert<int,T>(*extractSubDomain(boundaryCondition.getVoxelizedDomain().getVoxelMatrix(),vtkDomain)), "voxel", 1.);

    ImageWriter<T> imageWriter("leeloo");
    imageWriter.writeScaledPpm(fname, *boundaryCondition.computeVelocityNorm(imageDomain));
}

// This function produces images at predefined yz, xz and xy planes. The coordinates of the planes are given
//   in physical coordinates, and the output variables are velocity, vorticity and pressure.
void writeImages (
         OffLatticeBoundaryCondition3D<T,DESCRIPTOR,Velocity>& boundaryCondition, Array<T,3> location, T dx, T dt )
{
    plint nx = boundaryCondition.getLattice().getNx();
    plint ny = boundaryCondition.getLattice().getNy();
    plint nz = boundaryCondition.getLattice().getNz();
    Array<T,3> yz_plane(0.016960, 0.032604, 0.057772);
    Array<T,3> xz_plane(0.026725, 0.017978, 0.057772);
    Array<T,3> xy_plane(0.026725, 0.032604, 0.084113);

    Array<T,3> lyz_plane((yz_plane-location)/dx);
    Array<T,3> lxz_plane((xz_plane-location)/dx);
    Array<T,3> lxy_plane((xy_plane-location)/dx);

    Box3D yz_imageDomain (
            util::roundToInt(lyz_plane[0]), util::roundToInt(lyz_plane[0]),
            0, ny-1, 0, nz-1 );
    Box3D xz_imageDomain (
            0, nx-1,
            util::roundToInt(lxz_plane[1]), util::roundToInt(lxz_plane[1]),
            0, nz-1 );
    Box3D xy_imageDomain (
            0, nx-1, 0, ny-1,
            util::roundToInt(lxy_plane[2]), util::roundToInt(lxy_plane[2]) );

    Box3D yz_vtkDomain (
            util::roundToInt(lyz_plane[0])-3, util::roundToInt(lyz_plane[0])+3,
            0, ny-1, 0, nz-1 );
    Box3D xz_vtkDomain (
            0, nx-1,
            util::roundToInt(lxz_plane[1])-3, util::roundToInt(lxz_plane[1])+3,
            0, nz-1 );
    Box3D xy_vtkDomain (
            0, nx-1, 0, ny-1,
            util::roundToInt(lxy_plane[2])-3, util::roundToInt(lxy_plane[2])+3 );

    writeImages(boundaryCondition, xy_imageDomain, xy_vtkDomain, "xy", location, dx, dt);
    writeImages(boundaryCondition, xz_imageDomain, xz_vtkDomain, "xz", location, dx, dt);
    writeImages(boundaryCondition, yz_imageDomain, yz_vtkDomain, "yz", location, dx, dt);
}


// This is the function that prepares and performs the actual simulation.
void run ()
{
    plint margin = 3; // Extra margin of allocated cells around the obstacle. 
    plint borderWidth = 1; // Because the Guo boundary condition acts in a one-cell layer.
                           // Requirement: margin>=borderWidth.

    // The resolution is doubled at each coordinate direction with the increase of the
    //   resolution level by one. The parameter ``referenceResolution'' is by definition
    //   the resolution at grid refinement level 0.
    plint resolution = referenceResolution;

    // The next few lines of code are typical. They transform the surface geometry of the
    //   aneurysm given by the user to more efficient data structures that are internally
    //   used by palabos. The TriangleBoundary3D structure will be later used to assign
    //   proper boundary conditions.
    DEFscaledMesh<T>* defMesh =
        new DEFscaledMesh<T>(*triangleSet, resolution, referenceDirection, margin, extraLayer);
    TriangleBoundary3D<T> boundary(*defMesh);
    delete defMesh;
    boundary.getMesh().inflate();

    // When convective scaling is used (relationship of dt with respect to dx as the grid is
    //   refined) the value of the kinematic viscosity must be also properly adjusted.
    T nuLB_ = nuLB;
    T dx = boundary.getDx();
    T dt = nuLB_ / kinematicViscosity *dx*dx;
    T uAveLB = averageInletVelocity *dt/dx;
    T omega = 1./(3.*nuLB_+0.5);
    Array<T,3> location(boundary.getPhysicalLocation());


    pcout << "uLB=" << uAveLB << std::endl;
    pcout << "nuLB=" << nuLB_ << std::endl;
    pcout << "tau=" << 1./omega << std::endl;
    if (performOutput) {
        pcout << "dx=" << dx << std::endl;
        pcout << "dt=" << dt << std::endl;
    }

    // Next the inlets and outlets are identified (according to what the user has specified)
    //   in the input XML file, and proper boundary conditions are assigned.
    std::vector<BoundaryProfile3D<T,Velocity>*> inletOutlets;
    setOpenings(inletOutlets, boundary, uAveLB);
    Array<T,3> inletCenter(0.0, 0.0, 0.0);
    for (pluint i=0; i<openings.size(); ++i) {
        if (openings[i].inlet) {
            pcout << "Inner radius of inlet " << i << " : "
                  << openings[i].innerRadius << " lattice nodes" << std::endl;
            inletCenter=openings[i].center;
        }
    }
    T inletZpos = util::roundToInt(inletCenter[2])+1;
    BoundaryProfiles3D<T,Velocity> profiles;
    profiles.defineInletOutletTags(boundary, openingSortDirection);
    profiles.setInletOutlet(inletOutlets);

    // The aneurysm simulation is an interior (as opposed to exterior) flow problem. For
    //   this reason, the lattice nodes that lay inside the computational domain must
    //   be identified and distinguished from the ones that lay outside of it. This is
    //   handled by the following voxelization process.
    const int flowType = voxelFlag::inside;
    VoxelizedDomain3D<T> voxelizedDomain (
            boundary, flowType, extraLayer, borderWidth, extendedEnvelopeWidth, blockSize );

    pcout << getMultiBlockInfo(voxelizedDomain.getVoxelMatrix()) << std::endl;

    MultiScalarField3D<int> flagMatrix((MultiBlock3D&)voxelizedDomain.getVoxelMatrix());
    setToConstant(flagMatrix, voxelizedDomain.getVoxelMatrix(),
                  voxelFlag::inside, flagMatrix.getBoundingBox(), 1);
    setToConstant(flagMatrix, voxelizedDomain.getVoxelMatrix(),
                  voxelFlag::innerBorder, flagMatrix.getBoundingBox(), 1);
    plint numFluidCells = computeSum(flagMatrix);
    pcout << "Number of fluid cells: " << numFluidCells << std::endl;

    Dynamics<T,DESCRIPTOR>* dynamics = 0;
    if (useIncompressible) {
        dynamics = new IncBGKdynamics<T,DESCRIPTOR>(omega); // In this model velocity equals momentum.
    }
    else {
        dynamics = new BGKdynamics<T,DESCRIPTOR>(omega); // In this model velocity equals momentum
                                                         //   divided by density.
    }
    std::unique_ptr<MultiBlockLattice3D<T,DESCRIPTOR> > lattice 
        = generateMultiBlockLattice<T,DESCRIPTOR> (
                voxelizedDomain.getVoxelMatrix(), envelopeWidth, dynamics );
    lattice->toggleInternalStatistics(false);

    // The next piece of code is put for efficiency reasons at communications in parallel runs.
    //   The efficiency advantage comes essentially because the density and velocity are
    //   written in different fields.
    std::vector<MultiBlock3D*> rhoBarJarg;
    plint numScalars = 4;
    MultiNTensorField3D<T>* rhoBarJfield =
          generateMultiNTensorField3D<T>(*lattice, extendedEnvelopeWidth, numScalars);
    rhoBarJfield->toggleInternalStatistics(false);
    rhoBarJarg.push_back(rhoBarJfield);
    plint processorLevel=0;
    integrateProcessingFunctional (
            new PackedRhoBarJfunctional3D<T,DESCRIPTOR>(),
            lattice->getBoundingBox(), *lattice, *rhoBarJfield, processorLevel );

    // The Guo off lattice boundary condition is set up.
    GuoOffLatticeModel3D<T,DESCRIPTOR>* model =
            new GuoOffLatticeModel3D<T,DESCRIPTOR> (
                new TriangleFlowShape3D<T,Array<T,3> > (
                    voxelizedDomain.getBoundary(), profiles),
                flowType, useAllDirections );
    model->setVelIsJ(useIncompressible); // When the incompressible BGK model is used, velocity equals momentum.
    model->selectUseRegularizedModel(useRegularizedWall);
    model->selectComputeStat(false);
    OffLatticeBoundaryCondition3D<T,DESCRIPTOR,Velocity> boundaryCondition (
            model, voxelizedDomain, *lattice);
    boundaryCondition.insert(rhoBarJarg);

    iniLattice(*lattice, voxelizedDomain);
    
    global::timer("iteration").start();

    pcout << "Running " << numIter << " iteration." << endl;

    // Start PMT sensor
    std::unique_ptr<pmt::PMT> CPUsensor = pmt::rapl::Rapl::Create();
    pmt::State CPUstart = CPUsensor->Read();

    // Collision and streaming iterations.
    for(int i = 0; i < numIter; ++i)
    {
        if (i%200==0 && performOutput) {
            pcout << "T= " << currentTime << "; "
                  << "Average energy: "
                  << boundaryCondition.computeAverageEnergy()*util::sqr(dx/dt) << std::endl;
        }

        lattice->collideAndStream();
        currentTime = i*dt;
    }
    delete rhoBarJfield;

    double elapsed = global::timer("iteration").stop();
    pcout << "Elapsed: " << elapsed << " seconds" << endl;
    pcout << "Performance: " << (T)numFluidCells * (T)numIter / elapsed * 1.e-6 << " MLUPS" << endl;


    pmt::State CPUend = CPUsensor->Read();

    int nranks = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    pcout << "NAME: " << "Palabos_anerusym" << endl;
    pcout << "ALGO: "<< "LBM" << endl;
    pcout << "PRECISION: "<< sizeof (double) <<" bytes"<< endl;
    pcout << "OMP_THREADS: "<< 1 << std::endl;
    pcout << "MPI_RANKS: "<< nranks << std::endl;
    pcout << "NGPUs: "<< 0 << std::endl;
    pcout << "GPU ID: "<< 99 << std::endl;
    pcout << "SIZE: " << numFluidCells << endl;
    pcout << "(RAPL) CPU_TIME: " << pmt::PMT::seconds(CPUstart, CPUend) << " s"<< endl;
    pcout << "(RAPL) CPU_TIME_var: " << 0.0 << " s^2"<< endl;
    pcout << "(RAPL) CPU_TIME_std: " << 0.0 << " s"<< endl;
    pcout << "(RAPL) CPU_WATTS: " << pmt::PMT::watts(CPUstart, CPUend) << " W" << endl;
    pcout << "(RAPL) CPU_WATTS_var: " << 0.0 << " W^2" << endl;
    pcout << "(RAPL) CPU_WATTS_std: " << 0.0 << " W" << endl;
    pcout << "(RAPL) CPU_JOULES: " << pmt::PMT::joules(CPUstart, CPUend) << " J" << endl;
    pcout << "(RAPL) CPU_JOULES_var: " << 0.0 << " J^2" << endl;
    pcout << "(RAPL) CPU_JOULES_std: " << 0.0 << " J" << endl;
    pcout << "NRUNS: " << 1 << endl;


    Box3D measureBox(lattice->getBoundingBox());
    measureBox.z0=measureBox.z1=(plint)inletZpos;
    T inletPressure = DESCRIPTOR<T>::cs2*(boundaryCondition.computeAverageDensity(measureBox)-1.);

    // Image output.
    if (doImages) {
        writeImages(boundaryCondition, location, dx, dt);
        std::vector<std::string> scalarNames;
        scalarNames.push_back("pressure");
        scalarNames.push_back("wss");
        std::vector<T> scalarFactor;
        scalarFactor.push_back(util::sqr(dx/dt)*fluidDensity);
        scalarFactor.push_back(util::sqr(dx/dt)*fluidDensity);
        std::vector<std::string> vectorNames;
        vectorNames.push_back("force");
        std::vector<T> vectorFactor;
        vectorFactor.push_back(util::sqr(dx/dt)*fluidDensity);
        bool dynamicMesh = false;
        writeSurfaceVTK (
                boundary,
                *computeSurfaceForce( boundary, voxelizedDomain, *lattice, model->velIsJ(), dynamicMesh ),
                scalarNames, vectorNames, "surface.vtk", dynamicMesh, 0,
                scalarFactor, vectorFactor );
    }

    T averageEnergy = boundaryCondition.computeAverageEnergy()*util::sqr(dx/dt);
    T rmsVorticity  = boundaryCondition.computeRMSvorticity()/dt;
    T pressureDrop = inletPressure*util::sqr(dx/dt)*fluidDensity;
    T inletAverageVel = boundaryCondition.computeAverageVelocityComponent(measureBox,2)*dx/dt;

    if (performOutput) {
        pcout << "Average energy: " << averageEnergy << std::endl;
        pcout << "Total energy: " << averageEnergy*volume << std::endl;
        pcout << "RMS vorticity * volume * 0.5: " << rmsVorticity*0.5*volume << std::endl;
        pcout << "Pressure drop: " << pressureDrop << std::endl;
        pcout << "Average velocity through inlet section: " << inletAverageVel << std::endl;
        pcout << "Number of iterations: " << numIter << std::endl;
        pcout << averageEnergy*volume << ", " << pressureDrop << ", " << rmsVorticity*volume*0.5 << ", ";
    }

    if (performOutput) {
        std::vector<T> pointData = pointMeasures(*lattice, location, dx, dt);
        for (pluint i=0; i<pointData.size(); ++i) {
            pcout << pointData[i];
            if (i!=pointData.size()-1) {
                pcout << ", ";
            }
        }
        pcout << std::endl;
    }
}

// Read the user input XML file provided at the command-line.
void readParameters(XMLreader const& document)
{
    std::string meshFileName;
    std::vector<std::string> openingType;
    document["geometry"]["mesh"].read(meshFileName);
    document["geometry"]["inletDiameter"].read(userDefinedInletDiameter);
    document["geometry"]["averageInletVelocity"].read(averageInletVelocity);
    document["geometry"]["openings"]["sortDirection"].read(openingSortDirection);
    document["geometry"]["openings"]["type"].read(openingType);

    document["fluid"]["kinematicViscosity"].read(kinematicViscosity);
    document["fluid"]["density"].read(fluidDensity);
    document["fluid"]["volume"].read(volume);

    document["numerics"]["referenceDirection"].read(referenceDirection);
    document["numerics"]["referenceResolution"].read(referenceResolution);
    document["numerics"]["nuLB"].read(nuLB);

    document["simulation"]["numIter"].read(numIter);
    document["simulation"]["epsilon"].read(epsilon);

    document["simulation"]["doImages"].read(doImages);
    document["simulation"]["useAllDirections"].read(useAllDirections);
    document["simulation"]["useRegularizedWall"].read(useRegularizedWall);
    document["simulation"]["useIncompressible"].read(useIncompressible);
    document["simulation"]["poiseuilleInlet"].read(poiseuilleInlet);

    // At this part, the surface geometry of the aneurysm (as given by the user in
    //   the form of an ASCII or binary STL file) is read into a data structure
    //   comprised by a set of triangles. The DBL constant means that double
    //   precision accuracy will be used (generally the recommended choice).
    triangleSet = new TriangleSet<T>(meshFileName, DBL);
    pcout << "Reynolds number, based on provided inlet diameter: "
          << averageInletVelocity*userDefinedInletDiameter/kinematicViscosity
          << std::endl;
    plbIOError(openingSortDirection<0 || openingSortDirection>2,
               "Sort-direction of opening must be 0 (x), 1 (y), or 2 (z).");
    // The surface geometry, as provided by the STL file, must contain openings,
    //   namely inlets and outlets. On these openings, appropriate boundary conditions
    //   will be imposed by palabos. Which opening is inlet and which is outlet, is
    //   identified by the user in the input XML file.
    openings.resize(openingType.size());
    for (pluint i=0; i<openingType.size(); ++i) {
        std::string next_opening = util::tolower(openingType[i]);
        if (next_opening=="inlet") {
            openings[i].inlet = true;
        }
        else if (next_opening=="outlet") {
            openings[i].inlet = false;
        }
        else {
            plbIOError("Unknown opening type.");
        }
    }
}


int main(int argc, char* argv[])
{
    plbInit(&argc, &argv);
    global::directories().setOutputDir("./");
    global::IOpolicy().activateParallelIO(false);

    string paramXmlFileName;
    try {
        global::argv(1).read(paramXmlFileName);
    }
    catch (PlbIOException& exception) {
        pcout << "Wrong parameters; the syntax is: " 
              << (std::string)global::argv(0) << " parameter-input-file.xml" << std::endl;
        return -1;
    }

    // Read the parameter XML input file. (Lots of comments are included there too).
    try {
        XMLreader document(paramXmlFileName);
        readParameters(paramXmlFileName);
    }
    catch (PlbIOException& exception) {
        pcout << "Error in input file " << paramXmlFileName
              << ": " << exception.what() << std::endl;
        return -1;
    }

    global::timer("global").start();
    plint iniLevel=0;
    // This code incorporates the concept of smooth grid refinement until convergence is
    //   achieved. The word ``smooth'' indicates that as the refinement level increases
    //   by one, the whole grid doubles in each direction. When the grid is refined, both
    //   dx and dt have to change. Whether dt is changed as dx^2 (diffusive behavior)
    //   or as dx (convective behavior), is controlled by the input variable
    //   ``convectiveScaling'' (the recommended choice is not to use convective scaling).
    try {
        pcout << std::endl << "Running the simulation." << std::endl;
        run();
    }
    catch(PlbException& exception) {
        pcout << exception.what() << std::endl;
        return -1;
    }
}

