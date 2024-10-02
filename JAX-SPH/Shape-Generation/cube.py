import jax
import jax.numpy as jnp

def generate_cube_positions(spacing:   float,
                            edge_size: int):
    p = int(edge_size/spacing)
    x = jnp.repeat(jnp.repeat(jnp.linspace(0,edge_size,p),p),p)
    y = jnp.repeat(jnp.tile(jnp.linspace(0,edge_size,p),p),p)
    z = jnp.tile(jnp.tile(jnp.linspace(0,edge_size,p),p),p)
    xyz = jnp.stack([x,y,z],axis=1)
    return xyz