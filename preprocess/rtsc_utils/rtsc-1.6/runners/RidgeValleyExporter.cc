
#include <iostream>
#include "TriMesh.h"
#include "XForm.h"
#include "timestamp.h"
#ifndef DARWIN
#endif
#include <algorithm>
#include <regex>

using namespace trimesh;
using namespace std;


// Globals: mesh...
TriMesh *themesh;

xform xf;
char *xffilename; // Filename where we look for "home" position

fstream ridge_file;
fstream valley_file;

// Toggles for drawing various lines
int draw_ridges = 0, draw_valleys = 0, draw_apparent = 0;
int draw_K = 0, draw_H = 0, draw_DwKr = 0;
int draw_bdy = 0, draw_isoph = 0, draw_topo = 0;
int niso = 20, ntopo = 20;
float topo_offset = 0.0f;
int draw_faded = 1;

// Toggles for tests we perform
int test_c = 1, test_sc = 1, test_sh = 1, test_ph = 1, test_rv = 1, test_ar = 1;
float sug_thresh = 0.01, sh_thresh = 0.02, ph_thresh = 0.04;
float rv_thresh = 0.1, ar_thresh = 0.1;

// Other miscellaneous variables
float feature_size;	// Used to make thresholds dimensionless
float currsmooth;	// Used in smoothing
vec currcolor;		// Current line color



// Draw part of a ridge/valley curve on one triangle face.  v0,v1,v2
// are the indices of the 3 vertices; this function assumes that the
// curve connects points on the edges v0-v1 and v1-v2
// (or connects point on v0-v1 to center if to_center is true)
void draw_segment_ridge(int v0, int v1, int v2,
			float emax0, float emax1, float emax2,
			float kmax0, float kmax1, float kmax2,
			float thresh, bool to_center, bool do_ridge)
{
	// Interpolate to find ridge/valley line segment endpoints
	// in this triangle and the curvatures there
	float w10 = fabs(emax0) / (fabs(emax0) + fabs(emax1));
	float w01 = 1.0f - w10;
	point p01 = w01 * themesh->vertices[v0] + w10 * themesh->vertices[v1];
	float k01 = fabs(w01 * kmax0 + w10 * kmax1);

	point p12;
	float k12;
	if (to_center) {
		// Connect first point to center of triangle
		p12 = (themesh->vertices[v0] +
		       themesh->vertices[v1] +
		       themesh->vertices[v2]) / 3.0f;
		k12 = fabs(kmax0 + kmax1 + kmax2) / 3.0f;
	} else {
		// Connect first point to second one (on next edge)
		float w21 = fabs(emax1) / (fabs(emax1) + fabs(emax2));
		float w12 = 1.0f - w21;
		p12 = w12 * themesh->vertices[v1] + w21 * themesh->vertices[v2];
		k12 = fabs(w12 * kmax1 + w21 * kmax2);
	}

	// Don't draw below threshold
	k01 -= thresh;
	if (k01 < 0.0f)
		k01 = 0.0f;
	k12 -= thresh;
	if (k12 < 0.0f)
		k12 = 0.0f;

	// Skip lines that you can't see...
	if (k01 == 0.0f && k12 == 0.0f)
		return;

	// Fade lines
	if (draw_faded) {
		k01 /= (k01 + thresh);
		k12 /= (k12 + thresh);
	} else {
		k01 = k12 = 1.0f;
	}

    // Draw the line segment
    if (do_ridge) {
        ridge_file << p01.x << " " << p01.y << " " << p01.z << "\n";
        ridge_file << p12.x << " " << p12.y << " " << p12.z << "\n";
    } else {
        valley_file << p01.x << " " << p01.y << " " << p01.z << "\n";
        valley_file << p12.x << " " << p12.y << " " << p12.z << "\n";
    }

}


// Draw ridges or valleys (depending on do_ridge) in a triangle v0,v1,v2
// - uses ndotv for backface culling (enabled with do_bfcull)
// - do_test checks for curvature maxima/minina for ridges/valleys
//   (when off, it draws positive minima and negative maxima)
// Note: this computes ridges/valleys every time, instead of once at the
//   start (given they aren't view dependent, this is wasteful)
// Algorithm based on formulas of Ohtake et al., 2004.
void draw_face_ridges(int v0, int v1, int v2,
		      bool do_ridge,
		      const vector<float> &ndotv,
		      bool do_bfcull, bool do_test, float thresh)
{
	// Backface culling
	if (likely(do_bfcull &&
		   ndotv[v0] <= 0.0f && ndotv[v1] <= 0.0f && ndotv[v2] <= 0.0f))
		return;

	// Check if ridge possible at vertices just based on curvatures
	if (do_ridge) {
		if ((themesh->curv1[v0] <= 0.0f) ||
		    (themesh->curv1[v1] <= 0.0f) ||
		    (themesh->curv1[v2] <= 0.0f))
			return;
	} else {
		if ((themesh->curv1[v0] >= 0.0f) ||
		    (themesh->curv1[v1] >= 0.0f) ||
		    (themesh->curv1[v2] >= 0.0f))
			return;
	}

	// Sign of curvature on ridge/valley
	float rv_sign = do_ridge ? 1.0f : -1.0f;

	// The "tmax" are the principal directions of maximal curvature,
	// flipped to point in the direction in which the curvature
	// is increasing (decreasing for valleys).  Note that this
	// is a bit different from the notation in Ohtake et al.,
	// but the tests below are equivalent.
	const float &emax0 = themesh->dcurv[v0][0];
	const float &emax1 = themesh->dcurv[v1][0];
	const float &emax2 = themesh->dcurv[v2][0];
	vec tmax0 = rv_sign * themesh->dcurv[v0][0] * themesh->pdir1[v0];
	vec tmax1 = rv_sign * themesh->dcurv[v1][0] * themesh->pdir1[v1];
	vec tmax2 = rv_sign * themesh->dcurv[v2][0] * themesh->pdir1[v2];

	// We have a "zero crossing" if the tmaxes along an edge
	// point in opposite directions
	bool z01 = ((tmax0 DOT tmax1) <= 0.0f);
	bool z12 = ((tmax1 DOT tmax2) <= 0.0f);
	bool z20 = ((tmax2 DOT tmax0) <= 0.0f);

	if (z01 + z12 + z20 < 2)
		return;

	if (do_test) {
		const point &p0 = themesh->vertices[v0],
			    &p1 = themesh->vertices[v1],
			    &p2 = themesh->vertices[v2];

		// Check whether we have the correct flavor of extremum:
		// Is the curvature increasing along the edge?
		z01 = z01 && ((tmax0 DOT (p1 - p0)) >= 0.0f ||
			      (tmax1 DOT (p1 - p0)) <= 0.0f);
		z12 = z12 && ((tmax1 DOT (p2 - p1)) >= 0.0f ||
			      (tmax2 DOT (p2 - p1)) <= 0.0f);
		z20 = z20 && ((tmax2 DOT (p0 - p2)) >= 0.0f ||
			      (tmax0 DOT (p0 - p2)) <= 0.0f);

		if (z01 + z12 + z20 < 2)
			return;
	}

	// Draw line segment
	const float &kmax0 = themesh->curv1[v0];
	const float &kmax1 = themesh->curv1[v1];
	const float &kmax2 = themesh->curv1[v2];
	if (!z01) {
		draw_segment_ridge(v1, v2, v0,
				   emax1, emax2, emax0,
				   kmax1, kmax2, kmax0,
				   thresh, false, do_ridge);
	} else if (!z12) {
		draw_segment_ridge(v2, v0, v1,
				   emax2, emax0, emax1,
				   kmax2, kmax0, kmax1,
				   thresh, false, do_ridge);
	} else if (!z20) {
		draw_segment_ridge(v0, v1, v2,
				   emax0, emax1, emax2,
				   kmax0, kmax1, kmax2,
				   thresh, false, do_ridge);
	} else {
		// All three edges have crossings -- connect all to center
		draw_segment_ridge(v1, v2, v0,
				   emax1, emax2, emax0,
				   kmax1, kmax2, kmax0,
				   thresh, true, do_ridge);
		draw_segment_ridge(v2, v0, v1,
				   emax2, emax0, emax1,
				   kmax2, kmax0, kmax1,
				   thresh, true, do_ridge);
		draw_segment_ridge(v0, v1, v2,
				   emax0, emax1, emax2,
				   kmax0, kmax1, kmax2,
				   thresh, true, do_ridge);
	}
}


// Draw the ridges (valleys) of the mesh
void draw_mesh_ridges(bool do_ridge, const vector<float> &ndotv,
		      bool do_bfcull, bool do_test, float thresh)
{
	const int *t = &themesh->tstrips[0];
	const int *stripend = t;
	const int *end = t + themesh->tstrips.size();

	// Walk through triangle strips
	while (1) {
		if (unlikely(t >= stripend)) {
			if (unlikely(t >= end))
				return;
			// New strip: each strip is stored as
			// length followed by indices
			stripend = t + 1 + *t;
			// Skip over length plus first two indices of
			// first face
			t += 3;
		}

		draw_face_ridges(*(t-2), *(t-1), *t,
				 do_ridge, ndotv, do_bfcull, do_test, thresh);
		t++;
	}
}


// Compute a "feature size" for the mesh: computed as 1% of
// the reciprocal of the 10-th percentile curvature
void compute_feature_size()
{
	int nv = themesh->curv1.size();
	int nsamp = min(nv, 500);

	vector<float> samples;
	samples.reserve(nsamp * 2);

	for (int i = 0; i < nsamp; i++) {
		// Quick 'n dirty portable random number generator
		static unsigned randq = 0;
		randq = unsigned(1664525) * randq + unsigned(1013904223);

		int ind = randq % nv;
		samples.push_back(fabs(themesh->curv1[ind]));
		samples.push_back(fabs(themesh->curv2[ind]));
	}

	const float frac = 0.1f;
	const float mult = 0.01f;
	themesh->need_bsphere();
	float max_feature_size = 0.05f * themesh->bsphere.r;

	int which = int(frac * samples.size());
	nth_element(samples.begin(), samples.begin() + which, samples.end());

	feature_size = min(mult / samples[which], max_feature_size);
}


void usage(const char *myname)
{
	fprintf(stderr, "Usage: %s [-options] infile\n", myname);
	exit(1);
}


int main(int argc, char *argv[])
{
    if (argc < 2)
		usage(argv[0]);

	int i = 1;
	const char *filename = argv[i];
	std::string ridge_filename;
	std::string valley_filename;
	ridge_filename += filename;
	ridge_filename += ".ridge.txt";
	valley_filename += filename;
	valley_filename += ".valley.txt";

	themesh = TriMesh::read(filename);
	if (!themesh)
		usage(argv[0]);

	xffilename = new char[strlen(filename) + 4];
	strcpy(xffilename, filename);
	char *dot = strrchr(xffilename, '.');
	if (!dot)
		dot = strrchr(xffilename, 0);
	strcpy(dot, ".xf");

	themesh->need_tstrips();
	themesh->need_bsphere();
	themesh->need_normals();
	themesh->need_curvatures();
	themesh->need_dcurv();
	compute_feature_size();
	currsmooth = 0.5f * themesh->feature_size();

    static vector<float> ndotv, kr;

    ridge_file.open(ridge_filename, ios::out);
    if (!ridge_file) {
        cout << "Ridge file not created!";
    }
    else {
        draw_mesh_ridges(true, ndotv, false, test_rv,
                         rv_thresh / feature_size);
        ridge_file.close();
    }
    valley_file.open(valley_filename, ios::out);
    if (!valley_file) {
        cout << "Valley file not created!";
    }
    else {
        draw_mesh_ridges(false, ndotv, false, test_rv,
                         rv_thresh / feature_size);
        valley_file.close();
    }
}

