//
// Created by Seth Parker on 6/9/16.
// Angle-based Flattening implementation ported from the same in Blender
//

#ifndef VC_ABF_H
#define VC_ABF_H

#include <iostream>
#include <exception>

#include "itkQuadEdgeMeshBoundaryEdgesMeshFunction.h"

#include "vc_defines.h"
#include "vc_datatypes.h"
#include "deepCopy.h"

namespace volcart {
    namespace texturing {

      class abf {

      typedef itk::QuadEdgeMeshBoundaryEdgesMeshFunction< volcart::QuadMesh > BoundaryExtractor;

      public:
          ///// Constructors/Destructors /////
          abf(){};
          abf( VC_MeshType::Pointer input );
          ~abf(){};

          ///// Access Functions /////
          // Set inputs
          void setMesh( VC_MeshType::Pointer input );

          // Get outputs
          VC_MeshType::Pointer getMesh();
          volcart::UVMap getUVMap();

          ///// Process /////
          void compute();
      private:

          // Setup //
          void _fillQuadEdgeMesh();

          const VC_MeshType::Pointer  _mesh;
          volcart::QuadMesh::Pointer  _quadMesh;

          // Boundary and Interior Vertices
          // < id in quadMesh, id in list >
          std::map< volcart::QuadMeshPointIdentifier, volcart::QuadMeshPointIdentifier > _boundary;
          std::map< volcart::QuadMeshPointIdentifier, volcart::QuadMeshPointIdentifier > _interior;
      };

    }// texturing
}//volcart

#endif //VC_ABF_H
