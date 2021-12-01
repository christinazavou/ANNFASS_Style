
#define CGAL_EIGEN3_ENABLED
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/optimal_bounding_box.h>
#include <CGAL/Polygon_mesh_processing/triangulate_faces.h>
#include <CGAL/Polygon_mesh_processing/measure.h>
#include <CGAL/Real_timer.h>
#include <fstream>
#include <iostream>
#include <regex>

namespace PMP = CGAL::Polygon_mesh_processing;
typedef CGAL::Exact_predicates_inexact_constructions_kernel    K;
typedef K::Point_3                                             Point;
typedef CGAL::Surface_mesh<Point>                              Surface_mesh;

void run(const std::string& in_ply){

    std::string out_ply(regex_replace(in_ply, std::regex(".ply"), "_obb.ply"));

    std::ifstream input(in_ply);
    Surface_mesh sm;

    CGAL::read_ply(input, sm);

    CGAL::Real_timer timer;
    timer.start();
    // Compute the extreme points of the mesh, and then a tightly fitted oriented bounding box
    std::array<Point, 8> obb_points;
    CGAL::oriented_bounding_box(sm, obb_points,
                                CGAL::parameters::use_convex_hull(true));
    std::cout << "Elapsed time: " << timer.time() << std::endl;
    // Make a mesh out of the oriented bounding box
    Surface_mesh obb_sm;
    CGAL::make_hexahedron(obb_points[0], obb_points[1], obb_points[2], obb_points[3],
                          obb_points[4], obb_points[5], obb_points[6], obb_points[7], obb_sm);

    std::ofstream ofs;

    ofs.open (out_ply, std::ofstream::out);
    CGAL::write_ply(ofs, obb_sm);
    ofs.close();

//    PMP::triangulate_faces(obb_sm);
//    std::cout << "Volume: " << PMP::volume(obb_sm) << std::endl;

}
int main(int argc, char** argv)
{
//    TODO: check if file exists...
    std::string in_ply((argc > 1) ? argv[1] : "/home/christina/Documents/MARIASCAR.ply");
    run(in_ply);
}