import * as THREE from 'three';

export default function getStarfield({ numStars = 2000 } = {}) {
  const geometry = new THREE.BufferGeometry();
  const positions = [];

  for (let i = 0; i < numStars; i++) {
    const x = THREE.MathUtils.randFloatSpread(2000);
    const y = THREE.MathUtils.randFloatSpread(2000);
    const z = THREE.MathUtils.randFloatSpread(2000);
    positions.push(x, y, z);
  }

  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));

  const material = new THREE.PointsMaterial({ color: 0x888888 });
  return new THREE.Points(geometry, material);
}