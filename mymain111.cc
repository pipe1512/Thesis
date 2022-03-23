// main111.cc is a part of the PYTHIA event generator.
// Copyright (C) 2021 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Authors: Leif Lonnblad <leif.lonnblad@thep.lu.se>.

// Keywords: heavy ions; rivet; angantyr;

// This is a simple test program equivalent to main01.cc but using the
// Angantyr model for Heavy Ion collisions. It is still proton
// collisions, but uses the Angantyr impact parameter description to
// select collisions. It fits on one slide in a talk.  It studies the
// charged multiplicity distribution at the LHC.

// Optionally (by compiling with the flag -DRIVET and
// linking with rivet - see output of the command "rivet-config
// --cppflags --libs") it will send the event to Rivet for an ATLAS
// jet-analysis.

#include "Pythia8/Pythia.h"

#ifdef RIVET
#include "Pythia8/HeavyIons.h"
#include "Pythia8Plugins/Pythia8Rivet.h"
#endif

#include "Pythia8Plugins/ProgressLog.h"
#include <string>
#include <typeinfo>

using namespace Pythia8;

int main(int argc, char* argv[]) {
  // Generator. Process selection. LHC initialization. Histogram.
  Pythia pythia;


std::cout<< "Checking the args..." << "\n";

  std::string my_energy_arg;
  std::string sigma_D;
  std::string k_0;
  std::string alpha;
  std::string angantyr_model;

  my_energy_arg = argv[1];
  sigma_D = argv[2];
  k_0 = argv[3];
  alpha = argv[4];
  angantyr_model = argv[5];


  // First argument is always the name of your program
  std::cout << argv[0] << std::endl;
  std::cout << "Energy: " << argv[1] << std::endl;
  std::cout << "sigma_D: " << argv[2] << std::endl;
  std::cout << "k_0: " << argv[3] << std::endl;
  std::cout << "alpha: " << argv[4] << std::endl;
  std::cout << "HeavyIon:SigFitDefPar = " + sigma_D + "," + k_0 + "," + alpha + "," + "0.0,0.0,0.0,0.0,0.0" << std::endl;
  std::cout << "argc: " << argc << "\n";
  std::cout << "Running the program..." << "\n";

  pythia.readString("Beams:eCM = " + my_energy_arg + ".");
  pythia.readString("Angantyr:CollisionModel = " + angantyr_model + " ");
  pythia.readString("HeavyIon:mode = 2");
  pythia.readString("SoftQCD:all = on");
  if (std::stod(sigma_D) < 0 && std::stod(k_0) < 0 && std::stod(alpha) < 0){
    pythia.readString("HeavyIon:SigFitNGen = 0");
  } else {
    pythia.readString("HeavyIon:SigFitNGen = 20");
  }
  pythia.readString("HeavyIon:SigFitDefPar force= " + sigma_D + "," + k_0 + "," + alpha + "," + "0.0,0.0,0.0,0.0,0.0");
  pythia.init();


  return 0;
}

