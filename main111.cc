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
  std::string angantyr_model;
  std::string my_energy_arg;
  std::string seed_arg;

  std::cout << "argc: " << argc << "\n";
  std::cout << argv[0] << std::endl;
  my_energy_arg = argv[1];
  seed_arg = argv[2];
  angantyr_model = argv[3];
  std::cout << "1st Argument (energy) passed in is " << my_energy_arg << std::endl;
  std::cout << "2nd Argument (seed) passed in is " << seed_arg << std::endl;
  std::cout << "3rd Argument (Angantyr model) passed in is " << angantyr_model << std::endl;
  std::cout << "Running the program..." << "\n";


  pythia.readString("HeavyIon:SigFitNInt  = 1000000)");
  pythia.readString("Angantyr:CollisionModel = " + angantyr_model + " ");
  pythia.readString("HeavyIon:mode = 2");
  pythia.readString("Beams:eCM = " + my_energy_arg + ".");
  pythia.readString("SoftQCD:all = on");
  pythia.readString("HeavyIon:SigFitNGen = 20");
  pythia.rndm.init(stoi(seed_arg));
  pythia.init();

  return 0;
}

