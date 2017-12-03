

#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#if defined(__APPLE__)
#include <GLUT/GLUT.h>
#include <OpenGL/gl3.h>
#include <OpenGL/glu.h>
#else
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <windows.h>
#endif
#include <GL/glew.h>		 
#include <GL/freeglut.h>	
#endif

#include <vector>

#include "vec3.h"
#include "mat4x4.h"
#include "vec2.h"
#include "vec4.h"

const unsigned int windowWidth = 512, windowHeight = 512;

int majorVersion = 3, minorVersion = 0;

void getErrorInfo(unsigned int handle) 
{
	int logLen;
	glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &logLen);
	if (logLen > 0) 
	{
		char * log = new char[logLen];
		int written;
		glGetShaderInfoLog(handle, logLen, &written, log);
		printf("Shader log:\n%s", log);
		delete log;
	}
}

void checkShader(unsigned int shader, char * message) 
{
	int OK;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &OK);
	if (!OK) 
	{
		printf("%s!\n", message);
		getErrorInfo(shader);
	}
}

void checkLinking(unsigned int program) 
{
	int OK;
	glGetProgramiv(program, GL_LINK_STATUS, &OK);
	if (!OK) 
	{
		printf("Failed to link shader program!\n");
		getErrorInfo(program);
	}
}

class Shader
{
protected:
	unsigned int shaderProgram;

public:
	Shader()
	{
		const char *vertexSource = "\n\
			#version 410 \n\
			precision highp float; \n\
			\n\
			in vec2 vertexPosition;	\n\
			in vec2 vertexTexCoord; \n\
			out vec2 texCoord; \n\
			\n\
			void main() \n\
			{ \n\
				texCoord = vertexTexCoord; \n\
				gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1); \n\
			} \n\
		"; 

		const char *fragmentSource = "\n\
			#version 410 \n\
			precision highp float; \n\
			\n\
			uniform sampler2D samplerUnit; \n\
			in vec2 texCoord;  \n\
			out vec4 fragmentColor; \n\
			\n\
			void main() { \n\
			fragmentColor = texture(samplerUnit, texCoord);  \n\
			} \n\
		";

		unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
		if (!vertexShader) { printf("Error in vertex shader creation\n"); exit(1); }

		glShaderSource(vertexShader, 1, &vertexSource, NULL);
		glCompileShader(vertexShader);
		checkShader(vertexShader, "Vertex shader error");

		unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
		if (!fragmentShader) { printf("Error in fragment shader creation\n"); exit(1); }

		glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
		glCompileShader(fragmentShader);
		checkShader(fragmentShader, "Fragment shader error");

		shaderProgram = glCreateProgram();
		if (!shaderProgram) { printf("Error in shader program creation\n"); exit(1); }

		glAttachShader(shaderProgram, vertexShader);
		glAttachShader(shaderProgram, fragmentShader);

		glBindAttribLocation(shaderProgram, 0, "vertexPosition");
		glBindAttribLocation(shaderProgram, 1, "vertexTexCoord");

		glBindFragDataLocation(shaderProgram, 0, "fragmentColor");

		glLinkProgram(shaderProgram);
		checkLinking(shaderProgram);
	}

	~Shader()
	{
		if(shaderProgram) glDeleteProgram(shaderProgram);
	}

	void Run()
	{
		if(shaderProgram) glUseProgram(shaderProgram);
	}

	void UploadSamplerID()
	{
		int samplerUnit = 0; 
		int location = glGetUniformLocation(shaderProgram, "samplerUnit");
		glUniform1i(location, samplerUnit);
		glActiveTexture(GL_TEXTURE0 + samplerUnit); 
	}
};

Shader *shader = 0;

// Simple material class, with object color, and headlight shading.

class Material
{
    
    vec3 frontColor;
    vec3 backColor;
	vec3 kd;

public:
    
    Material(vec3 front, vec3 back){
        frontColor = front;
        backColor = back;
    }
    
	virtual vec3 getColor(
		vec3 position,
		vec3 normal,
		vec3 viewDir)
	{
        if( viewDir.dot(normal) > 0){
            return frontColor * viewDir.dot(normal);
        } else {
            return backColor * -viewDir.dot(normal);
        }
	}

	virtual vec3 shade(vec3 position,
		vec3 normal,
		vec3 viewDir,
		vec3 lightDir,
		vec3 powerDensity) {
		float cosTheta = normal.dot(lightDir);
		if (cosTheta < 0) return backColor * -normal.dot(viewDir) * -cosTheta * powerDensity;
		return getColor(position, normal, viewDir) * powerDensity * cosTheta;

	}
	//virtual vec3 shade(vec3 position, vec3 normal, vec3 lightDir, vec3 powerDensity) {
	//	float cosTheta = normal.dot(lightDir);
	//	if (cosTheta < 0) return vec3(0, 0, 0);
	//	return kd * powerDensity * cosTheta;

	//	//return lightDir;
	//}

	float snoise(vec3 r) {
		unsigned int x = 0x0625DF73;
		unsigned int y = 0xD1B84B45;
		unsigned int z = 0x152AD8D0;
		float f = 0;
		for (int i = 0; i<32; i++) {
			vec3 s(x / (float)0xffffffff,
				y / (float)0xffffffff,
				z / (float)0xffffffff);
			f += sin(s.dot(r));
			x = x << 1 | x >> 31;
			y = y << 1 | y >> 31;
			z = z << 1 | z >> 31;
		}
		return f / 64.0 + 0.5;
	}

};

class Ebony : public Material
{
	float shininess;

public:
	Ebony() : Material(vec3(0.3, 0.3, 0.7), vec3(0.3, 0.3, 0.7))
	{
		shininess = 10;
	}
public:
	virtual vec3 shade(vec3 position,
		vec3 normal,
		vec3 viewDir,
		vec3 lightDir,
		vec3 powerDensity) {
		vec3 kd = getColor(position, normal, viewDir);
		vec3 ks = vec3(.850, .85, .85);
		vec3 diffuseLight;
		vec3 specLight;
		float cosTheta = normal.dot(lightDir);
		vec3 halfway = (viewDir + lightDir).normalize();
		float cosDelta = normal.dot(halfway);

		if (cosTheta < 0)
		{
			diffuseLight = vec3(0, 0, 0);
			specLight = vec3(0, 0, 0);
		}
		else if (cosDelta < 0)
		{
			specLight = vec3(0, 0, 0);
			diffuseLight = kd * powerDensity * cosTheta;
		}
		else
		{
			specLight = powerDensity * ks * pow(cosDelta, shininess);
			diffuseLight = kd * powerDensity * cosTheta;
		}

		return specLight + diffuseLight;

	}


};

//class SpecularMaterial : public Material {
//	vec3 ks;
//	vec3 kd;
//	float shininess;
//
//public:
//	SpecularMaterial(vec3 color, vec3 kd, vec3 ks, float shininess)
//		: Material(color) {
//		this->ks = ks;
//		this->shininess = shininess;
//		this->kd = kd;
//	}
//	vec3 getColor(vec3 position,
//		vec3 normal, vec3 viewDir, vec3 lightDir,
//		vec3 powerDensity) {
//
//		if (viewDir.dot(normal) > 0) {
//			return frontColor
//				* shade(lightDir, normal, powerDensity, viewDir);
//		}
//		else {
//			return backColor
//				* shade(lightDir, normal, powerDensity, viewDir);
//		}
//	}
//
//	float shade(vec3 lightDir, vec3 normal, vec3 powerDensity, vec3 viewDir) {
//		vec3 H = (lightDir + viewDir.normalize()).normalize();
//		float res = powerDensity.dot(kd) * max(normal.dot(lightDir), 0) + powerDensity.dot(ks) * pow(max(normal.dot(H), 0), shininess);
//		return res;
//	}
//};

class Wood : public Material
{
	float scale;
	float turbulence;
	float period;
	float sharpness;
public:
	Wood(): Material(vec3(1, 1, 1), vec3(1, 1, 1))
	{
		scale = 16;
		turbulence = 500;
		period = 8;
		sharpness = 10;
	}
	virtual vec3 getColor(
		vec3 position,
		vec3 normal,
		vec3 viewDir)
	{
		//return normal;
		float w = position.x * period + pow(snoise(position * scale), sharpness)*turbulence + 10000.0;
		w -= int(w);

		int x = ((position.x + 2) * 2);
		int y = ((position.z + 2) * 2);

		if ((x + y) % 2 == 0)

			return (vec3(1, 0.3, 0) * w + vec3(0.35, 0.1, 0.05) * (1 - w)) * normal.dot(viewDir);


		else return vec3(0, 0, 0);

	}
};

class Marble : public Material
{
	float scale;
	float turbulence;
	float period;
	float sharpness;
public:
	Marble() :
		Material(vec3(1, 1, 1), vec3(1, 1, 1))
	{
		scale = 32;
		turbulence = 50;
		period = 32;
		sharpness = 1;
	}
	virtual vec3 getColor(
		vec3 position,
		vec3 normal,
		vec3 viewDir)
	{

		float w = position.x * period + pow(snoise(position * scale), sharpness)*turbulence;
		w = pow(sin(w)*0.5 + 0.5, 4);

		
		return (vec3(0, 0, 1) * w + vec3(1, 1, 1) * (1 - w)) * normal.dot(viewDir);

		//return normal;
		
	}
};

class ChessBoard : public Material
{

public:
	ChessBoard() : Material(vec3(1, 1, 1), vec3(1, 1, 1))
	{

	}
	virtual vec3 getColor(
		vec3 position,
		vec3 normal,
		vec3 viewDir)
	{
		int x = ((position.x + 2) * 2);
		int y = ((position.z + 2) * 2);

		if ((x + y) % 2 == 0) 
	 
			return vec3(1, 0, 0); 


		else return vec3(0, 0, 0); 
	}
};


//class SolidTexture: Material{
//    vec3 frontColor;
//    vec3 backColor;
//public:
//    SolidTexture(vec3 front, vec3 back){
//        
//    }
//};

// Camera class.

class Camera
{
	vec3 eye;		// World space camera position.
	vec3 lookAt;	// Center of window in world space.
	vec3 right;		// Vector from window center to window right-mid (in world space).
	vec3 up;		// Vector from window center to window top-mid (in world space).

public:
	Camera()
	{
		eye = vec3(0, 0, 2);
		lookAt = vec3(0, 0, 1);
		right = vec3(1, 0, 0);
		up = vec3(0, 1, 0);
	}
	vec3 getEye()
	{
		return eye;
	}

	// Compute ray through pixel at normalized device coordinates.

	vec3 rayDirFromNdc(float x, float y) {
		return (lookAt - eye
			+ right * x
			+ up    * y
			).normalize();
	}
};

// Ray structure.

class Ray
{
public:
    vec3 origin;
    vec3 dir;
    Ray(vec3 o, vec3 d)
    {
        origin = o;
        dir = d;
    }
};

// Hit record structure. Contains all data that describes a ray-object intersection point.

class Hit
{
public:
	Hit()
	{
		t = -1;
	}
	float t;				// Ray paramter at intersection. Negative means no valid intersection.
	vec3 position;			// Intersection coordinates.
	vec3 normal;			// Surface normal at intersection.
	Material* material;		// Material of intersected surface.
};

// Abstract base class.

class Intersectable
{
protected:
	Material* material;
public:
	Intersectable(Material* material):material(material) {}
    virtual Hit intersect(const Ray& ray)=0;
};

// Simple helper class to solve quadratic equations with the Quadratic Formula [-b +- sqrt(b^2-4ac)] / 2a, and store the results.

class QuadraticRoots
{
public:
	float t1;
	float t2;

	// Solves the quadratic a*t*t + b*t + c = 0 using the Quadratic Formula [-b +- sqrt(b^2-4ac)] / 2a, and sets members t1 and t2 to store the roots.

	QuadraticRoots(float a, float b, float c)
	{
        float discr = b * b - 4.0 * a * c;
        if ( discr < 0 ) // no roots
		{
			t1 = -1;
			t2 = -1;
			return;
		}
        float sqrt_discr = sqrt( discr );
		t1 = (-b + sqrt_discr)/2.0/a;
		t2 = (-b - sqrt_discr)/2.0/a;
	}

	// Returns the lesser of the positive solutions, or a negative value if there was no positive solution.

	float getLesserPositive()
	{
		return (0 < t1 && (t2 < 0 || t1 < t2)) ? t1 : t2;
	}
};

// Object realization.

class Sphere : public Intersectable
{
	vec3 center;
	float radius;
public:
    Sphere(const vec3& center, float radius, Material* material):
		Intersectable(material),
		center(center),
		radius(radius)
    {
    }
	QuadraticRoots solveQuadratic(const Ray& ray)
	{
        vec3 diff = ray.origin - center;
        float a = ray.dir.dot(ray.dir);
        float b = diff.dot(ray.dir) * 2.0;
        float c = diff.dot(diff) - radius * radius;
		return QuadraticRoots(a, b, c);
 
	}
	vec3 getNormalAt(vec3 r)
	{
		return (r - center).normalize();
	}
    Hit intersect(const Ray& ray)
    {
		// This is a generic intersect that works for any shape with a quadratic equation. solveQuadratic should solve the proper equation (+ ray equation) for the shape, and getNormalAt should return the proper normal.
        
        float t = solveQuadratic(ray).getLesserPositive();
			
		Hit hit;
		hit.t = t;
		hit.material = material;
		hit.position = ray.origin + ray.dir * t;
		hit.normal = getNormalAt(hit.position);

		return hit;
    }
}; 

//class Plane : public Intersectable
//{
//    vec3 n;
//    vec3 r0;
//public:
//    Plane(const vec3& normal, vec3 r, Material* material):
//    Intersectable(material),
//    n(normal),
//    r0(r)
//    {
//    }
//
//    vec3 getNormalAt(vec3 r)
//    {
//        return n;
//    }
//    
//    Hit intersect(const Ray& ray)
//    {
//
//        float t = (r0 - ray.origin).dot(n) / ray.dir.dot(n);
//        
//        Hit hit;
//        hit.t = t;
//        hit.material = material;
//        hit.position = ray.origin + ray.dir * t;
//        hit.normal = getNormalAt(hit.position);
//        
//        return hit;
//    }
//};

class Plane : public Intersectable
{
	vec3 n;
	vec3 r0;
public:
	Plane(const vec3& normal, vec3 r, Material* material) :
		Intersectable(material),
		n(normal),
		r0(r)
	{
	}

	vec3 getNormalAt(vec3 r)
	{
		return n;
	}

	Hit intersect(const Ray& ray)
	{


		float t = (r0 - ray.origin).dot(n) / ray.dir.dot(n);

		Hit hit;
		hit.t = t;
		hit.material = material;
		hit.position = ray.origin + ray.dir * t;
		hit.normal = getNormalAt(hit.position);

		if (hit.position.x >= 2 || hit.position.x <= -2 || hit.position.z >= 2 || hit.position.z <= -2) hit.t = -1;

		return hit;
	}
};

class DiffuseMaterial
{
    
    vec3 frontColor;
    vec3 backColor;
	vec3 kd;
    
public:
    
    DiffuseMaterial(vec3 front, vec3 back){
        frontColor = front;
        backColor = back;
    }
    
    virtual vec3 getColor(
                          vec3 position,
                          vec3 normal,
                          vec3 viewDir)
    {
        if( viewDir.dot(normal) > 0){
            return frontColor * viewDir.dot(normal);
        } else {
            return backColor * -viewDir.dot(normal);
        }
    }
    
    virtual vec3 shade(vec3 normal, vec3 lightDir, vec3 powerDensity){
		float cosTheta = normal.dot(lightDir);
		if (cosTheta < 0) return vec3(0, 0, 0);
		return kd * powerDensity * cosTheta;

        //return lightDir;
    }
    
    
};

class LightSource
{
    
    
public:
    
    vec3 powerDensity;
    
    virtual vec3 getPowerDensityAt(vec3 x) = 0;
    virtual vec3 getLightDirAt(vec3 x) = 0;
    virtual float getDistanceFrom(vec3 x) = 0;
    
};

class DirectionalLight : public LightSource
{
    vec3 lightDir;
    
public:
    DirectionalLight(vec3 dir, vec3 col = vec3(0.2,0.2,0.2)){
        lightDir = dir;
        powerDensity = col;
    }
    
    vec3 getPowerDensityAt(vec3 x){
        return powerDensity;
    }
    
    vec3 getLightDirAt(vec3 x){
        return lightDir;
    }
    
    float getDistanceFrom(vec3 x){
        return 1000000.0;
    }
    
    
    
};

class PointLight : public LightSource
{
	vec3 lightPos;

public:
	PointLight(vec3 pos, vec3 col = vec3(0.2, 0.2, 0.2)) {
		lightPos = pos;
		powerDensity = col;
	}

	vec3 getPowerDensityAt(vec3 x) {

		float n = 1 / (getDistanceFrom(x) * getDistanceFrom(x));

		return powerDensity*n;
	}

	float getDistanceFrom(vec3 x) {
		return (x - lightPos).norm();
	}

	vec3 getLightDirAt(vec3 x) {
		return lightPos - x;
	}

};

class Quadric: public Intersectable
{
    mat4x4 coeffs;
    
public:
    Quadric(Material* material):
    Intersectable(material)
    {
        
        coeffs = *new mat4x4(0,0,0,0,
                            0,0,0,0,
                            0,0,0,0,
                            0,0,0,0);
        
        
    }
    
    Quadric* sphere(){
        coeffs = *new mat4x4(1,0,0,0,
                             0,1,0,0,
                             0,0,1,0,
                             0,0,0,-1);
        
        return this;
    }
    
    
    Quadric* cylinder(){
        coeffs = *new mat4x4(1,0,0,0,  //x
                             0,0,0,0,  //z
                             0,0,1,0,  //y
                             0,0,0,-1);
        
        return this;
    }
    
    Quadric* cone(){
        coeffs = *new mat4x4(1,0,0,0,  //x
                             0,-1,0,0,  //z
                             0,0,1,0,  //y
                             0,0,0,0);
        
        return this;
    }
    
	Quadric* paraboloid() {
		coeffs = *new mat4x4(1, 0, 0, 0,  //x
			0, 0, 0, -1,  //z
			0, 0, 1, 0,  //y
			0, 0, 0, 0);

		return this;
	}

	Quadric* hyperbolicParaboloid() {
		coeffs = *new mat4x4(1, 0, 0, 0,  //x
			0, 0, 0, -1,  //z
			0, 0, -1, 0,  //y
			0, 0, 0, 0);

		return this;
	}


	Quadric* hyperboloid() {
		coeffs = *new mat4x4(1, 0, 0, 0,  //x
			0, -1, 0, 0,  //z
			0, 0, 1, 0,  //y
			0, 0, 0, -1);

		return this;
	}


    
    
    
    QuadraticRoots solveQuadratic(const Ray& ray)
    {
        vec4 d = vec4(ray.dir);
        vec4 e = vec4(ray.origin);
        d.v[3] = 0;
        float a = d.dot(coeffs*d);
        float b = d.dot(coeffs*e) + (e.dot(coeffs*d));
        float c = e.dot(coeffs*e);
        return QuadraticRoots(a, b, c);
        
    }
    
    vec3 getNormalAt(vec3 r)
    {
        vec4 vec = coeffs*r + operator*(vec4(r), coeffs) ;
        return vec3(vec.x,vec.y,vec.z).normalize();
    }
    
    Hit intersect(const Ray& ray)
    {
        // This is a generic intersect that works for any shape with a quadratic equation. solveQuadratic should solve the proper equation (+ ray equation) for the shape, and getNormalAt should return the proper normal.
        
        float t = solveQuadratic(ray).getLesserPositive();
        
        Hit hit;
        hit.t = t;
        hit.material = material;
        hit.position = ray.origin + ray.dir * t;
        hit.normal = getNormalAt(hit.position);
        
        return hit;
    }
    
    
    Quadric* transform(mat4x4 t){
        coeffs = t.invert() * coeffs * t.invert().transpose();
        return this;
    }
    
    bool contains(vec3 r)
    {
        vec4 rhomo(r);
        
        if(rhomo.dot(coeffs*rhomo)<0){
            return true;
        }
        
        return false;
    }
    
    Quadric* parallelPlanes(float height) {
        coeffs._00 = 0;
        coeffs._11 = 1;
        coeffs._22 = 0;
        coeffs._33 = -(height/2)*(height/2);
        
        return this;
    }
    
	Quadric* PlanesLeftRright(float width) {
		coeffs._00 = 1;
		coeffs._11 = 0;
		coeffs._22 = 0;
		coeffs._33 = -(width / 2)*(width / 2);

		return this;
	}

	Quadric* PlanesFrontBack(float width) {
		coeffs._00 = 0;
		coeffs._11 = 0;
		coeffs._22 = 1;
		coeffs._33 = -(width / 2)*(width / 2);

		return this;
	}
};


class ClippedQuadric : public Intersectable
{
    Quadric shape;
    Quadric clipper;
    
public:
    
    ClippedQuadric(Material* material):
    Intersectable(material), shape(*new Quadric(material)), clipper(*new Quadric(material))
    {
    }
    
    
    Hit intersect(const Ray& ray){
        
        QuadraticRoots root = shape.solveQuadratic(ray);
        vec3 position1 = ray.origin + ray.dir * root.t1;
        vec3 position2 = ray.origin + ray.dir * root.t2;
        
        if(!clipper.contains(position1)){
            root.t1 = -1;
        }
        
        if(!clipper.contains(position2)){
            root.t2 = -1;
        }
        
        
        float t = root.getLesserPositive();
        
        Hit hit;
        hit.t = t;
        hit.material = material;
        hit.position = ray.origin + ray.dir * t;
        hit.normal = shape.getNormalAt(hit.position);
        
        return hit;
    }
    
    
    ClippedQuadric* transform(mat4x4 m){
        
        shape = *shape.transform(m);
        clipper = *clipper.transform(m);
        
        return this;
    }
    
    ClippedQuadric* cylinder(float height){
        shape = *shape.cylinder();
        clipper = *clipper.parallelPlanes(height);
        
        return this;
    }

    
    ClippedQuadric* sphere(float height){
        shape = *shape.sphere();
        clipper = *clipper.parallelPlanes(height);
        return this;
    }
    
    
	ClippedQuadric* cone(float height) {
		shape = *shape.cone();
		clipper = *clipper.parallelPlanes(height)->transform(mat4x4::translation(vec3(0, -height / 2, 0)));
		return this;
	}

	ClippedQuadric* paraboloid(float height) {
		shape = *shape.paraboloid();
		clipper = *clipper.parallelPlanes(height / 2)->transform(mat4x4::translation(vec3(0, height / 4, 0)));
		return this;
	}


	ClippedQuadric* hyperboloid(float height) {
		shape = *shape.hyperboloid();
		clipper = *clipper.parallelPlanes(height)->transform(mat4x4::translation(vec3(0, -height / 4, 0)));
		return this;
	}

	ClippedQuadric* sphereClippedSphere() {
		shape = *shape.sphere();
		clipper = *clipper.cylinder()
			->transform(mat4x4::scaling(vec3(1, 0.5, 1))*
				mat4x4::rotation(vec3(1, 0, 0), M_PI/2)*
				mat4x4::translation(vec3(0.5, 0.5, 0)));
		return this;
	}
	
	ClippedQuadric* bishopInside() {
		shape = *shape.sphere()->transform(
			mat4x4::scaling(vec3(2, 0.5, 2))*
			mat4x4::translation(vec3(2, 0.4, 0)));
		clipper = *clipper.sphere();
		return this;
	}

	ClippedQuadric* box(float height) {
		shape = *shape.parallelPlanes(2);
		clipper = *clipper.parallelPlanes(height);
		return this;
	}

    
};

class MultiClippedQuadric : public Intersectable
{
	Quadric shape;
	Quadric clipper1;
	Quadric clipper2;
	Quadric clipper3;

public:

	MultiClippedQuadric(Material* material) :
		Intersectable(material), shape(*new Quadric(material)), clipper1(*new Quadric(material)), clipper2(*new Quadric(material)), clipper3(*new Quadric(material))
	{
	}


	Hit intersect(const Ray& ray) {

		QuadraticRoots root = shape.solveQuadratic(ray);
		vec3 position1 = ray.origin + ray.dir * root.t1;
		vec3 position2 = ray.origin + ray.dir * root.t2;

		if (!clipper1.contains(position1)) {
			root.t1 = -1;
		}


		if (!clipper1.contains(position2)) {
			root.t2 = -1;
		}


		if (!(clipper2.contains(position1) || clipper3.contains(position1))) {
			root.t1 = -1;
		}

		if (!(clipper2.contains(position2) || clipper3.contains(position2))) {
			root.t2 = -1;
		}





		float t = root.getLesserPositive();

		Hit hit;
		hit.t = t;
		hit.material = material;
		hit.position = ray.origin + ray.dir * t;
		hit.normal = shape.getNormalAt(hit.position);

		return hit;
	}



	MultiClippedQuadric* transform(mat4x4 m) {

		shape = *shape.transform(m);
		clipper1 = *clipper1.transform(m);
		clipper2 = *clipper2.transform(m);
		clipper3 = *clipper3.transform(m);

		return this;
	}

	MultiClippedQuadric* cylinder(float height) {
		shape = *shape.cylinder();

		clipper1 = *clipper1.parallelPlanes(height);
		clipper2 = *clipper2.PlanesFrontBack(0.6);
		clipper3 = *clipper3.PlanesLeftRright(0.6);

		return this;
	}

};

class OppositeClippedQuadric : public Intersectable
{
	Quadric shape;
	Quadric clipper;

public:

	OppositeClippedQuadric(Material* material) :
		Intersectable(material), shape(*new Quadric(material)), clipper(*new Quadric(material))
	{
	}


	Hit intersect(const Ray& ray) {

		QuadraticRoots root = shape.solveQuadratic(ray);
		vec3 position1 = ray.origin + ray.dir * root.t1;
		vec3 position2 = ray.origin + ray.dir * root.t2;

		if (clipper.contains(position1)) {
			root.t1 = -1;
		}

		if (clipper.contains(position2)) {
			root.t2 = -1;
		}


		float t = root.getLesserPositive();

		Hit hit;
		hit.t = t;
		hit.material = material;
		hit.position = ray.origin + ray.dir * t;
		hit.normal = shape.getNormalAt(hit.position);

		return hit;
	}


	OppositeClippedQuadric* transform(mat4x4 m) {

		shape = *shape.transform(m);
		clipper = *clipper.transform(m);

		return this;
	}


	OppositeClippedQuadric* bishopTop() {
		shape = *shape.sphere();
		clipper = *clipper.sphere()->transform(mat4x4::scaling(vec3(2, 0.5, 2))*
			mat4x4::translation(vec3(2, 0.4, 0)));
		return this;
	}


};

class ClippedQuadricInside : public Intersectable
{
	Quadric shape;
	Quadric clipper;

public:

	ClippedQuadricInside(Material* material) :
		Intersectable(material), shape(*new Quadric(material)), clipper(*new Quadric(material))
	{
	}


	Hit intersect(const Ray& ray) {

		QuadraticRoots root = shape.solveQuadratic(ray);
		vec3 position1 = ray.origin + ray.dir * root.t1;
		vec3 position2 = ray.origin + ray.dir * root.t2;

		if (!clipper.contains(position1)) {
			root.t1 = -1;
		}

		if (!clipper.contains(position2)) {
			root.t2 = -1;
		}


		float t = root.getLesserPositive();

		Hit hit;
		hit.t = t;
		hit.material = material;
		hit.position = ray.origin + ray.dir * t;
		hit.normal = -shape.getNormalAt(hit.position);

		return hit;
	}


	ClippedQuadricInside* transform(mat4x4 m) {

		shape = *shape.transform(m);
		clipper = *clipper.transform(m);

		return this;
	}


	ClippedQuadricInside* bishopInside() {
		shape = *shape.sphere()->transform(
			mat4x4::scaling(vec3(2, 0.5, 2))*
			mat4x4::translation(vec3(2, 0.4, 0)));
		clipper = *clipper.sphere();
		return this;
	}

};

class Pawns : public Intersectable
{
	Quadric sphere;
	ClippedQuadric cone;

public:

	Pawns(Material* material) :
		Intersectable(material), sphere(*new Quadric(material)), cone(*new ClippedQuadric(material))
	{
		sphere = *sphere.sphere()
			->transform(mat4x4::scaling(vec3(0.15, 0.15, 0.15))*
				mat4x4::translation(vec3(0, 0.05, -0.02)));

		cone = *cone.cone(1.5)
			->transform(mat4x4::scaling(vec3(0.2, 0.6, 0.2)));

	}


	Hit intersect(const Ray& ray) {

		Hit coneHit = cone.intersect(ray);
		Hit sphereHit = sphere.intersect(ray);

		if (coneHit.t < 0) return sphereHit;
		if (sphereHit.t < 0) return coneHit;

		return sphereHit.t < coneHit.t ? sphereHit : coneHit;

	}


	Pawns* transform(mat4x4 m) {

		sphere = *sphere.transform(m);
		cone = *cone.transform(m);

		return this;
	}


};

class Bishop : public Intersectable
{
	ClippedQuadric* body;
	OppositeClippedQuadric* top;
	ClippedQuadricInside* inside;

public:

	Bishop(Material* material) :
		Intersectable(material), top(new OppositeClippedQuadric(material)), body(new ClippedQuadric(material)), inside(new ClippedQuadricInside(material))
	{

		top = top->bishopTop()
			->transform(mat4x4::scaling(vec3(0.2, 0.2, 0.2)) *
				mat4x4::translation(vec3(0, 0, 0)));

		inside = inside->bishopInside()
			->transform(mat4x4::scaling(vec3(0.2, 0.2, 0.2)) *
				mat4x4::translation(vec3(0, 0, 0)));

		body = body->cone(1.5)
			->transform(mat4x4::scaling(vec3(0.2, 0.6, 0.2)) *
				mat4x4::translation(vec3(0, 0, 0)));


	}



	Hit intersect(const Ray& ray) {

		Hit topHit = top->intersect(ray);
		Hit insideHit = inside->intersect(ray);
		Hit bodyHit = body->intersect(ray);

		Hit dummyHit;
		dummyHit.t = 10000;

		if (topHit.t > 0 && topHit.t < dummyHit.t) dummyHit = topHit;

		if (insideHit.t > 0 && insideHit.t < dummyHit.t) dummyHit = insideHit;

		if (bodyHit.t > 0 && bodyHit.t < dummyHit.t) dummyHit = bodyHit;


		dummyHit.t = dummyHit.t == 10000 ? -1 : dummyHit.t;


		return dummyHit;

	}


	Bishop* transform(mat4x4 m) {

		top = top->transform(m);
		inside = inside->transform(m);
		body = body->transform(m);

		return this;
	}


};

class King : public Intersectable
{
	ClippedQuadric body;
	Quadric cross1;
	Quadric cross2;
	ClippedQuadric crown;

public:

	King(Material* material) :
		Intersectable(material), crown(*new ClippedQuadric(material)), body(*new ClippedQuadric(material)), cross1(*new Quadric(material)), cross2(*new Quadric(material))
	{

		cross1 = *cross1.sphere()
			->transform(mat4x4::scaling(vec3(0.05, 0.2, 0.05)) *
				mat4x4::translation(vec3(0, 0.22, 0)));

		cross2 = *cross2.sphere()
			->transform(mat4x4::scaling(vec3(0.2, 0.05, 0.05)) *
				mat4x4::translation(vec3(0, 0.25, 0)));

		body = *body.hyperboloid(1.5)
			->transform(mat4x4::scaling(vec3(0.15, 0.6, 0.15)) *
				mat4x4::translation(vec3(0, -0.3, 0)));


		crown = *crown.paraboloid(1)
			->transform(mat4x4::scaling(vec3(0.4, 0.6, 0.4))*
				mat4x4::translation(vec3(0, -0.2, 0)));

	}


	Hit intersect(const Ray& ray) {

		Hit crossHit1 = cross1.intersect(ray);
		Hit crossHit2 = cross2.intersect(ray);
		Hit crownHit = crown.intersect(ray);
		Hit bodyHit = body.intersect(ray);

		Hit dummyHit;
		dummyHit.t = 10000;

		if (crossHit1.t > 0 && crossHit1.t < dummyHit.t) dummyHit = crossHit1;

		if (crossHit2.t > 0 && crossHit2.t < dummyHit.t) dummyHit = crossHit2;

		if (crownHit.t > 0 && crownHit.t < dummyHit.t) dummyHit = crownHit;

		if (bodyHit.t > 0 && bodyHit.t < dummyHit.t) dummyHit = bodyHit;

		dummyHit.t = dummyHit.t == 10000 ? -1 : dummyHit.t;


		return dummyHit;

	}


	King* transform(mat4x4 m) {

		cross1 = *cross1.transform(m);
		cross2 = *cross2.transform(m);
		crown = *crown.transform(m);
		body = *body.transform(m);

		return this;
	}


};


class Queen : public Intersectable
{
	ClippedQuadric body;
	Quadric top;
	Quadric head;
	Quadric neck;

public:

	Queen(Material* material) :
		Intersectable(material), top(*new Quadric(material)), body(*new ClippedQuadric(material)), neck(*new Quadric(material)), head(*new Quadric(material))
	{

		top = *top.sphere()
			->transform(mat4x4::scaling(vec3(0.05, 0.05, 0.05)) *
				mat4x4::translation(vec3(0, 0.03, 0.01)));

		head = *head.sphere()
			->transform(mat4x4::scaling(vec3(0.16, 0.1, 0.16)) *
				mat4x4::translation(vec3(0, -0.1, 0)));

		body = *body.hyperboloid(1.4)
			->transform(mat4x4::scaling(vec3(0.15, 0.6, 0.15)) *
				mat4x4::translation(vec3(0, -0.3, 0)));


		neck = *neck.sphere()
			->transform(mat4x4::scaling(vec3(0.2, 0.06, 0.2))*
				mat4x4::translation(vec3(0, -0.2, 0)));

	}

	float reg(vec3 r) {
		unsigned int x = 2;
		unsigned int y = 2;
		unsigned int z = 3;
		float f = 0;
		for (int i = 0; i<32; i++) {
			vec3 s (1/x,
				1/y,
				1/z);
			f += sin(s.dot(r));
			x = x << 1 | x >> 31;
			y = y << 1 | y >> 31;
			z = z << 1 | z >> 31;
		}
		return f / 64.0 + 0.5;
	}

	/*float snoise(vec3 r) {
		r.x = cos(r.x);
		r.y = cos(r.y);
		r.z = cos(r.z);

		return r;
	}*/


	float dress(vec3 pos)
	{
		float scale = 50;
		float turbulence = 1000;
		float period = 1;
		float sharpness = 8;

		float w = pos.x * period + pow(reg(pos * scale), sharpness)*turbulence + 10000.0;
		//float w = pos.x * pos.z + pow(cos(pos.y * 20), sharpness) + 1000000.0;
		int k = int(w);

		int x = ((pos.x + 2) * 2);
		int y = ((pos.y + 2) * 2);

		if (k % 2 == 0)

			return 1;


		else return -1;


	}


	Hit intersect(const Ray& ray) {

		Hit topHit = top.intersect(ray);
		Hit headHit = head.intersect(ray);
		Hit bodyHit = body.intersect(ray);
		Hit neckHit = neck.intersect(ray);

		Hit dummyHit;
		dummyHit.t = 10000;

		if (topHit.t > 0 && topHit.t < dummyHit.t) dummyHit = topHit;

		if (headHit.t > 0 && headHit.t < dummyHit.t) dummyHit = headHit;

		if (bodyHit.t > 0 && bodyHit.t < dummyHit.t && dress(bodyHit.position) > 0) dummyHit = bodyHit;

		if (neckHit.t > 0 && neckHit.t < dummyHit.t) dummyHit = neckHit;

		dummyHit.t = dummyHit.t == 10000 ? -1 : dummyHit.t;


		return dummyHit;

	}


	Queen* transform(mat4x4 m) {

		top = *top.transform(m);
		head = *head.transform(m);
		body = *body.transform(m);
		neck = *neck.transform(m);

		return this;
	}


};

class Rook : public Intersectable
{
	ClippedQuadric body;
	ClippedQuadric head;
	MultiClippedQuadric top;

public:

	Rook(Material* material) :
		Intersectable(material), body(*new ClippedQuadric(material)), head(*new ClippedQuadric(material)), top(*new MultiClippedQuadric(material))
	{

		top = *top.cylinder(0.6)
			->transform(mat4x4::scaling(vec3(0.15, 0.15, 0.15))*
				mat4x4::translation(vec3(0, -0.06, 0)));

		head = *head.cylinder(0.3)
			->transform(mat4x4::scaling(vec3(0.15, 0.15, 0.15))*
				mat4x4::translation(vec3(0, -0.1, 0)));


		body = *body.hyperboloid(1.4)
			->transform(mat4x4::scaling(vec3(0.15, 0.6, 0.15)) *
				mat4x4::translation(vec3(0, -0.3, 0)));


	}


	Hit intersect(const Ray& ray) {

		Hit topHit = top.intersect(ray);
		Hit bodyHit = body.intersect(ray);
		Hit headHit = head.intersect(ray);

		Hit dummyHit;

		dummyHit.t = 10000;

		if (topHit.t > 0 && topHit.t < dummyHit.t) dummyHit = topHit;

		if (bodyHit.t > 0 && bodyHit.t < dummyHit.t) dummyHit = bodyHit;

		if (headHit.t > 0 && headHit.t < dummyHit.t) dummyHit = headHit;


		dummyHit.t = dummyHit.t == 10000 ? -1 : dummyHit.t;


		return dummyHit;

	}


	Rook* transform(mat4x4 m) {

		top = *top.transform(m);

		body = *body.transform(m);

		head = *head.transform(m);


		return this;
	}


};


class Scene
{
	Camera camera;
	std::vector<Intersectable*> objects;
	std::vector<Material*> materials;
    std::vector<LightSource*> lights;
public:
	Scene()
	{
		// BUILD YOUR SCENE HERE
        
		LightSource *point1 = new PointLight(vec3(0.5, 0.3, 0), vec3(3, 0.5, 0.5));
        LightSource *directional1 = new DirectionalLight(vec3(3, 1, 0) , vec3(0.5, 0.5, 0.5));
        
        lights.push_back(directional1);
		lights.push_back(point1);
        
        Material *material1 = new Material(vec3(0.7,0.2,0.5),vec3(0,1,0));
        Material *material2 = new Material(vec3(0.2,0.9,0.1),vec3(0,1,0));
        Material *material3 = new Material(vec3(0.3,0.3,0.7),vec3(0.3, 0.3, 0.7));
        materials.push_back(material1);
        materials.push_back(material2);
        materials.push_back(material3);
        
        /*Material *planeM = new Material(vec3(0.4,0.3,0.4), vec3(0,1,0));
        materials.push_back(new Wood());
        Plane *plane = new Plane(vec3(0,1,0), vec3(0,-1,0),materials[3]);
        objects.push_back(plane);*/

		Material *chessBoard = new Wood();
		materials.push_back(chessBoard);

		Plane *plane = new Plane(vec3(0, 1, 0), vec3(0, -1, 0), materials[3]);
		objects.push_back(plane);
        
        Material *quadricM = new Material(vec3(0.6,0.8,0.6), vec3(0,1,0));
        materials.push_back(quadricM);

		Material *ebony = new Ebony();
		materials.push_back(ebony);
        
        /*ClippedQuadric *q = new ClippedQuadric(materials[4]);
        *q = *q->cylinder(2)
        ->transform(
                mat4x4::rotation(vec3(1.5,0,0),1)*
                mat4x4::scaling(vec3(0.5,0.5,0.5))*
                mat4x4::translation(vec3(-0.5,0,0.5))
            );*/
        
        /*ClippedQuadric *cone = new ClippedQuadric(materials[2]);
        *cone = *cone->cone(2)
        ->transform(
                    mat4x4::rotation(vec3(0,0,0),1)*
                    mat4x4::scaling(vec3(0.1,0.15,0.1))*
                    mat4x4::translation(vec3(0.5,-0.3,0.5))
                    );

		ClippedQuadric *sphere = new ClippedQuadric(materials[2]);
		*sphere = *sphere->sphere(2)->transform(
			mat4x4::rotation(vec3(0, 0, 0), 1)*
			mat4x4::scaling(vec3(0.13, 0.13, 0.13))*
			mat4x4::translation(vec3(0.5, -0.2, 0.5))
		);*/

		Pawns *pawn = new Pawns(materials[5]);
		*pawn = *pawn->transform(mat4x4::translation(vec3(-0.4, 0, -0.1)));
		Pawns *pawn2 = new Pawns(materials[5]);
		*pawn2 = *pawn2->transform(mat4x4::translation(vec3(-0.7, 0, 0.3)));

		Bishop *bishop = new Bishop(materials[2]);
		*bishop = *bishop->transform(mat4x4::rotation(vec3(0,0,0), 1)*
			mat4x4::translation(vec3(-0.9, 0, 0.8)));

		Bishop *bishop1 = new Bishop(materials[2]);
		*bishop1 = *bishop1->transform(mat4x4::rotation(vec3(0, -1, 0), 1)*
			mat4x4::translation(vec3(0.9, 0, 0.8)));

		King *king = new King(materials[2]);
		*king = *king->transform(mat4x4::translation(vec3(0.8, 0, 0.3)));

		Queen *queen = new Queen(materials[2]);
		*queen = *queen->transform(mat4x4::translation(vec3(0.5, 0, 0)));

		Rook *rook = new Rook(materials[2]);
		rook = rook->transform(mat4x4::translation(vec3(0.1, 0, -0.5)));
		objects.push_back(rook);

		/*ClippedQuadric *crown = new ClippedQuadric(materials[2]);
		*crown = *crown->paraboloid(2)->transform(
			mat4x4::rotation(vec3(0, 0, 0), 1)*
			mat4x4::scaling(vec3(0.15, 0.15, 0.15))*
			mat4x4::translation(vec3(0.5, 0.18, 0.5))
		);*/

		/*ClippedQuadric *bishop = new ClippedQuadric(new Wood());
		*bishop = *bishop->bishop(2)->transform(
			mat4x4::rotation(vec3(1.5, 0, 0), 1)*
			mat4x4::scaling(vec3(0.5, 0.5, 0.5))*
			mat4x4::translation(vec3(-0.5, 0, 0.5))
		);*/
        
		/*ClippedQuadric *box = new ClippedQuadric(materials[2]);
		*box = *box->box(2)->transform(
			mat4x4::rotation(vec3(0, 0, 0), 1)*
			mat4x4::scaling(vec3(0.5, 0.5, 0.5))*
			mat4x4::translation(vec3(0, 0, 0))
		);*/
        
        //objects.push_back(q);
        //objects.push_back(cone);
		//objects.push_back(sphere);
		objects.push_back(pawn);
		objects.push_back(pawn2);
		objects.push_back(bishop);
		objects.push_back(king);
		objects.push_back(queen);
		objects.push_back(bishop1);
		//objects.push_back(crown);
		//objects.push_back(box);
        
        
	}
	~Scene()
	{
		 //UNCOMMENT THESE WHEN APPROPRIATE
		for (std::vector<Material*>::iterator iMaterial = materials.begin(); iMaterial != materials.end(); ++iMaterial)
			delete *iMaterial;
		for (std::vector<Intersectable*>::iterator iObject = objects.begin(); iObject != objects.end(); ++iObject)
			delete *iObject;		
	}

public:
	Camera& getCamera()
	{
		return camera;
	}
    
    Hit firstIntersect(Ray ray){
        float t = 10000;
        Hit hit;
        for(int i = 0; i <= objects.size() - 1; i++){
            Hit curHit = objects[i]->intersect(ray);
            if(curHit.t < t && curHit.t > 0){
                hit = curHit;
                t = curHit.t;
            }
        }
        
        return hit;
    }

	vec3 trace(const Ray& ray)
	{
		Hit hit = firstIntersect(ray);

		if (hit.t < 0)
		{
			vec3 rayDir = ray.dir;
			return vec3(.25 * cos(rayDir.x), .1 *cos(rayDir.z), .5 *sin(rayDir.y));
		}
        
        /*if(hit.t < 0)
            return vec3(1, 1, 1);*/
        
        vec3 color = hit.material->shade(hit.position, hit.normal, -ray.dir, lights[0]->getLightDirAt(hit.position), lights[0]->getPowerDensityAt(hit.position));

		
        
        
        /*float r = 0;
        float g = 0;
        float b = 0;
        
        if(lights.size() != 0){
            for(int i = 0; i <= lights.size() - 1; i++){
                float dist = lights[i]->getDistanceFrom(ray.origin);
                vec3 pd = lights[i]->getPowerDensityAt(ray.origin);
                r = fmax(pd.x, r);
                g = fmax(pd.y, g);
                b = fmax(pd.z, b);
            }
        }
        
        color.x = fmin(r,color.x);
        color.y = fmin(g,color.y);
        color.z = fmin(b,color.z);*/

		float epsilon = 0.01;
		        
		for(int i = 0; i < lights.size(); i++)
		 {
		   vec3 lightDir = lights[i]->getLightDirAt(hit.position);
		   Ray shadowRay(hit.position + hit.normal * epsilon, lightDir);
		        
		   Hit shadowHit = firstIntersect(shadowRay);
		        
		   if((shadowHit.t < 0.0) || (lights[i]->getDistanceFrom(hit.position) < shadowHit.t))
		      {
		        color += hit.material->shade(hit.position, hit.normal, -ray.dir, lightDir, lights[i]->getPowerDensityAt(hit.position));
		           }
		        }

		return color;
		
			//Hit hit = firstIntersect(ray);

			//if (hit.t < 0)
			//	return vec3(0, 0, 1);
			//if (hit.t == FLT_MAX) {
			//	vec3 rayDir = ray.dir;
			//	return vec3(.5 * cos(rayDir.x) + .5 * tan(rayDir.x), 0, sin(rayDir.y));
			//}

			//vec3 color;

			//float epsilon = .15;

			//for (LightSource* light : lights) {
			//	vec3 lightDir = light->getLightDirAt(hit.position);
			//	Ray shadowRay(hit.position + hit.normal * epsilon, lightDir);

			//	Hit shadowHit = firstIntersect(shadowRay);

			//	if (shadowHit.t < 0.0 || light->getDistanceFrom(hit.position) < shadowHit.t) {
			//		color += hit.material->getColor(hit.position, hit.normal, -ray.dir,
			//			light->getLightDirAt(hit.position), light->getPowerDensityAt(hit.position));
			//	}
			//}

			//return color;
			////return hit.material->getColor(hit.position, hit.normal, -ray.dir);
		}
    
};

Scene scene;




class FrameBuffer {
	unsigned int textureId;
	vec3 image[windowWidth * windowHeight];

public:
	FrameBuffer() {
		for(int i = 0; i < windowWidth * windowHeight; i++) image[i] = vec3(0.0, 0.0, 0.0);

		glGenTextures(1, &textureId); 
		glBindTexture(GL_TEXTURE_2D, textureId); 

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, windowWidth, windowHeight, 0, GL_RGB, GL_FLOAT, image);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); 
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); 
	}

	void Bind(Shader* s)
	{
		s->UploadSamplerID();
		glBindTexture(GL_TEXTURE_2D, textureId);
		
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, windowWidth, windowHeight, 0, GL_RGB, GL_FLOAT, image);
	}

	bool ComputeImage()
	{
		static unsigned int iPart = 0;

		if(iPart >= 64)
			return false;
		for(int j = iPart; j < windowHeight; j+=64)
		{
			for(int i = 0; i < windowWidth; i++)
			{
				float ndcX = (2.0 * i - windowWidth) / windowWidth;
				float ndcY = (2.0 * j - windowHeight) / windowHeight;
				Camera& camera = scene.getCamera();
				Ray ray = Ray(camera.getEye(), camera.rayDirFromNdc(ndcX, ndcY));
			
				image[j*windowWidth + i] = scene.trace(ray);
			}
		}
		iPart++;
		return true;
	}
};

class Screen {
	FrameBuffer frameBuffer;
	unsigned int vao;	

public:
	Screen() 
	{ 
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		unsigned int vbo[2];
		glGenBuffers(2, &vbo[0]);

		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]); 
		static float vertexCoords[] = { -1, -1,		1, -1,		-1, 1,		1, 1 };

		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW); 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);   

		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]); 
		static float vertexTextureCoords[] = { 0, 0,	1, 0,		0, 1,		1, 1 };

		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexTextureCoords), vertexTextureCoords, GL_STATIC_DRAW);
		glEnableVertexAttribArray(1);  
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, NULL); 
	}

	void Draw(Shader* s)
	{
		if(frameBuffer.ComputeImage())
		glutPostRedisplay();

		s->Run();
		frameBuffer.Bind(s);

		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glBindVertexArray(vao); 
		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
		glDisable(GL_BLEND);
	}
};

Screen *screen = 0;


void onDisplay( ) {
    glClearColor(0.1f, 0.2f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 

	screen->Draw(shader);

    glutSwapBuffers(); 
}

void onInitialization() 
{
	glViewport(0, 0, windowWidth, windowHeight);

	shader = new Shader();
	
	screen = new Screen();
}

void onExit() 
{
	delete screen; screen = 0;
	delete shader; shader = 0;	
	printf("exit");
}

int main(int argc, char * argv[]) {
	glutInit(&argc, argv);
#if !defined(__APPLE__)
	glutInitContextVersion(majorVersion, minorVersion);
#endif
	glutInitWindowSize(windowWidth, windowHeight);				
	glutInitWindowPosition(100, 100);							
#if defined(__APPLE__)
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_3_2_CORE_PROFILE);  
#else
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
	glutCreateWindow("Ray Casting");

#if !defined(__APPLE__)
	glewExperimental = true;	
	glewInit();
#endif

	printf("GL Vendor    : %s\n", glGetString(GL_VENDOR));
	printf("GL Renderer  : %s\n", glGetString(GL_RENDERER));
	printf("GL Version (string)  : %s\n", glGetString(GL_VERSION));
	glGetIntegerv(GL_MAJOR_VERSION, &majorVersion);
	glGetIntegerv(GL_MINOR_VERSION, &minorVersion);
	printf("GL Version (integer) : %d.%d\n", majorVersion, minorVersion);
	printf("GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

	glViewport(0, 0, windowWidth, windowHeight);

	onInitialization();

	glutDisplayFunc(onDisplay);                

	glutMainLoop();
		
	onExit();
	
	return 1;
}

