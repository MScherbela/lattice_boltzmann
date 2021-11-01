import numpy as np
import matplotlib.pyplot as plt

def naca_00(t, thickness=15):
    return 0.05*thickness*(0.2969*np.sqrt(t) - 0.1260*t - 0.3516*t**2 + 0.2843*t**3 - 0.1036*t**4)

def naca_airfoil(x, camber, camber_pos, thickness):
    m = camber / 100
    p = camber_pos / 10
    yc = np.where(x < p, m / p ** 2 * (2 * p * x - x ** 2), m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x ** 2))
    dyc_dx = np.where(x < p, 2 * m / p ** 2 * (p - x), 2 * m / (1 - p ** 2) * (p - x))
    theta = np.arctan(dyc_dx)

    y00 = naca_00(x, thickness)
    xu = x - y00 * np.sin(theta)
    xl = x + y00 * np.sin(theta)
    yu = yc + y00 * np.cos(theta)
    yl = yc - y00 * np.cos(theta)
    return xu, yu, xl, yl

def rotate(degrees, x0, y0, x, y):
    phi = degrees * np.pi / 180
    M = np.array([[np.cos(phi), np.sin(phi)],[-np.sin(phi), np.cos(phi)]])
    offset = np.array([x0, y0])[:,None]
    z = np.stack([x,y], axis=0) - offset
    z = M @ z + offset
    return z[0,:], z[1,:]


def to_raster_image(nx, ny, xu, yu, xl, yl, x_offset, y_offset, scale):
    xu = np.array(np.round(xu*scale), dtype=int)
    yu = np.array(np.round(yu*scale), dtype=int)
    xl = np.array(np.round(xl*scale), dtype=int)
    yl = np.array(np.round(yl*scale), dtype=int)

    img = np.zeros([ny, nx], dtype=bool)
    img[y_offset - yu, x_offset + xu] = 1
    img[y_offset - yl, x_offset + xl] = 1
    for col in range(nx):
        ind = np.where(img[:,col])[0]
        if len(ind) >= 2:
            img[min(ind):max(ind), col] = 1
    return img

def get_airfoil_mask(nx, ny, offset_x, offset_y, angle, scale, camber=4, camber_pos=4, thickness=15):
    x = np.linspace(0, 1, 1000)
    xu, yu, xl, yl = naca_airfoil(x, camber, camber_pos, thickness)
    xu, yu = rotate(angle, 0.5, 0, xu, yu)
    xl, yl = rotate(angle, 0.5, 0, xl, yl)
    img = to_raster_image(nx, ny, xu, yu, xl, yl, offset_x, offset_y, scale)
    return img

if __name__ == '__main__':
    degrees = 30
    x = np.linspace(0,1,500)
    xu, yu, xl, yl = naca_airfoil(x, 4,4,15)
    xu, yu = rotate(degrees, 0.5, 0, xu, yu)
    xl, yl = rotate(degrees, 0.5, 0, xl, yl)

    img = to_raster_image(400,300, xu, yu, xl, yl, 50, 150, 200.0)

    plt.close("all")
    plt.subplot(1,2,1)
    plt.plot(xu, yu, color='k')
    plt.plot(xl, yl, color='k')
    plt.axis("equal")

    plt.subplot(1,2,2)
    plt.imshow(img)

