import React, { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

const Starfield = () => {
  const starGroup = useRef();

  useFrame(() => {
    starGroup.current.rotation.y += 0.0002; // Slight rotation for the starfield
  });

  const stars = new Array(2000).fill().map(() => {
    const x = THREE.MathUtils.randFloatSpread(500); // Random star position
    const y = THREE.MathUtils.randFloatSpread(500);
    const z = THREE.MathUtils.randFloatSpread(500);
    return [x, y, z];
  });

  return (
    <group ref={starGroup}>
      {stars.map((star, i) => (
        <mesh key={i} position={star}>
          <sphereGeometry args={[0.1, 12, 12]} /> {/* Make stars smaller */}
          <meshBasicMaterial color="white" />
        </mesh>
      ))}
    </group>
  );
};

export default Starfield;