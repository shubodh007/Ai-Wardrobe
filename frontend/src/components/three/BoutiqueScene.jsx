import { useRef } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { 
  Environment, 
  Float, 
  MeshTransmissionMaterial, 
  Sparkles, 
  MeshDistortMaterial,
  BakeShadows
} from '@react-three/drei'
import * as THREE from 'three'

// The central "AI Lens" or "Vision Orb"
function VisionOrb({ scrollY }) {
  const mesh = useRef()
  
  useFrame((state, delta) => {
    const scroll = scrollY ? scrollY.get() : 0
    
    // Slow continuous rotation
    mesh.current.rotation.y += delta * 0.1
    mesh.current.rotation.x += delta * 0.05
    
    // Scroll-based parallax & transformation
    // As you scroll down (scroll -> 1), orb moves to the right and back
    const targetX = THREE.MathUtils.lerp(0, 3.5, scroll)
    const targetY = THREE.MathUtils.lerp(0, 1.5, scroll)
    const targetZ = THREE.MathUtils.lerp(0, -2, scroll)
    
    mesh.current.position.x = THREE.MathUtils.damp(mesh.current.position.x, targetX, 4, delta)
    mesh.current.position.y = THREE.MathUtils.damp(mesh.current.position.y, targetY, 4, delta)
    mesh.current.position.z = THREE.MathUtils.damp(mesh.current.position.z, targetZ, 4, delta)
  })

  return (
    <Float floatIntensity={1.5} speed={1.5} text="AI Vision">
      <mesh ref={mesh} scale={2.2}>
        <sphereGeometry args={[1, 64, 64]} />
        <MeshTransmissionMaterial 
          backside={true}
          samples={4}
          thickness={1.5}
          chromaticAberration={0.03}
          anisotropy={0.1}
          distortion={0.4}
          distortionScale={0.3}
          temporalDistortion={0.05}
          color="#ffffff"
          transmission={1}
          roughness={0.05}
          ior={1.5}
        />
      </mesh>
    </Float>
  )
}

// "Digital Fabric" abstract ribbon 
function SyntheticFabric({ scrollY }) {
  const mesh = useRef()
  const material = useRef()
  
  useFrame((state, delta) => {
    const scroll = scrollY ? scrollY.get() : 0
    
    mesh.current.rotation.y = THREE.MathUtils.damp(mesh.current.rotation.y, scroll * Math.PI, 3, delta)
    mesh.current.rotation.x = THREE.MathUtils.damp(mesh.current.rotation.x, scroll * Math.PI * 0.5, 3, delta)
    
    // Move from bottom left to center-left as we scroll down
    const targetX = THREE.MathUtils.lerp(-3, -2, scroll)
    const targetY = THREE.MathUtils.lerp(-2, 1, scroll)
    const targetZ = THREE.MathUtils.lerp(-1, 2, scroll)
    
    mesh.current.position.x = THREE.MathUtils.damp(mesh.current.position.x, targetX, 3, delta)
    mesh.current.position.y = THREE.MathUtils.damp(mesh.current.position.y, targetY, 3, delta)
    mesh.current.position.z = THREE.MathUtils.damp(mesh.current.position.z, targetZ, 3, delta)
    
    if (material.current) {
        material.current.distort = THREE.MathUtils.lerp(0.2, 0.5, scroll)
    }
  })

  return (
    <Float floatIntensity={3} speed={2} rotationIntensity={1}>
      <mesh ref={mesh} position={[-3, -2, -1]}>
        <torusKnotGeometry args={[1.5, 0.4, 200, 32]} />
        <MeshDistortMaterial 
          ref={material}
          color="#a78bfa"
          roughness={0.2}
          metalness={0.8}
          distort={0.2}
          speed={2}
          emissive="#7b61ff"
          emissiveIntensity={0.1}
        />
      </mesh>
    </Float>
  )
}

// Secondary floating accents
function AbstractAccent({ scrollY, initialPos, color, speedFn }) {
    const mesh = useRef()
    
    useFrame((state, delta) => {
        const scroll = scrollY ? scrollY.get() : 0
        mesh.current.rotation.x += delta * 0.2
        mesh.current.rotation.y += delta * 0.3
        
        // Custom parallax
        const scrollParallax = speedFn(scroll)
        mesh.current.position.y = THREE.MathUtils.damp(
            mesh.current.position.y,
            initialPos[1] + scrollParallax, 
            4, 
            delta
        )
    })
    
    return (
        <Float floatIntensity={2} speed={1.5}>
            <mesh ref={mesh} position={initialPos} scale={0.4}>
                <octahedronGeometry args={[1, 0]} />
                <meshStandardMaterial color={color} roughness={0.1} metalness={0.9} wireframe />
            </mesh>
        </Float>
    )
}

export default function BoutiqueScene({ scrollYProgress }) {
  return (
    <div className="absolute inset-0 z-0 pointer-events-none">
      <Canvas 
        camera={{ position: [0, 0, 8], fov: 45 }}
        dpr={[1, 2]} // Support high-DPI displays for crisp rendering
      >
        <ambientLight intensity={0.4} />
        <directionalLight position={[5, 10, 5]} intensity={1.5} color="#ffffff" />
        <directionalLight position={[-5, -10, -5]} intensity={0.8} color="#a78bfa" />
        <directionalLight position={[10, 0, -10]} intensity={0.5} color="#f3e5b3" />
        
        <VisionOrb scrollY={scrollYProgress} />
        <SyntheticFabric scrollY={scrollYProgress} />
        
        <AbstractAccent 
          scrollY={scrollYProgress} 
          initialPos={[4, -3, 0]} 
          color="#e8c547" 
          speedFn={(s) => s * 6} 
        />
        <AbstractAccent 
          scrollY={scrollYProgress} 
          initialPos={[-4, 2, -2]} 
          color="#4ad6e2" 
          speedFn={(s) => s * -4} 
        />
        
        <Sparkles count={150} scale={15} size={1.5} speed={0.3} color="#f3e5b3" opacity={0.3} />
        
        <Environment preset="city" />
        <BakeShadows />
      </Canvas>
    </div>
  )
}
