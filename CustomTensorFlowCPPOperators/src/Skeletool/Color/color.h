//==============================================================================================//
// Classname:
//      color
//
//==============================================================================================//
// Description:
//		This class defines a color with its value and color space. This
//		class implements color distance function that takes into account
//		different color spaces and helps not mixing them.
//
//==============================================================================================//

#pragma once 

//==============================================================================================//

#include <Eigen/Core>
#include <vector>
#include <iostream>
#include <math.h>

//==============================================================================================//

// known color spaces
enum ColorSpace
{
	RGB, // in [0..1]
	HSV,
	HCV,
	CGZ,
	CUSTOM_MARKER_DETECTION,
	RGC, // rg-Chromaticity (r g G)
	COLOR_GRADIENT_TENSOR
};

//==============================================================================================//

class Color
{
    public:

		// constructor: both value and color space are
		// required to be initialized
		Color(const Eigen::Vector3f& v, const ColorSpace& s):value(v),cs(s) {};

		// constructor: by default, black is set
		Color() { Black(); };

		// returns black in the RGB format
		static Color Black() { return Color(Eigen::Vector3f(0, 0, 0), RGB); }

		// returns white in the RGB format
		static Color White() { return Color(Eigen::Vector3f(1.0f, 1.0f, 1.0f), RGB); }

		// returns red in the RGB format
		static Color Red() { return Color(Eigen::Vector3f(1.0f, 0.0f, 0.0f), RGB); }

		// returns green in the RGB format
		static Color Green() { return Color(Eigen::Vector3f(0.0f, 1.0f, 0.0f), RGB); }

		// returns blue in the RGB format
		static Color Blue() { return Color(Eigen::Vector3f(0.0f, 0.0f, 1.0f), RGB); }

		// returns gray in the RGB format
		static Color Gray() { return Color(Eigen::Vector3f(0.6f, 0.6f, 0.6f), RGB); }

		// get the color value
		const Eigen::Vector3f getValue() const { return value; }
		Eigen::Vector4f getRGBAValue();

		// the the color value in the specified color
		// space (no change to the current color)
		Eigen::Vector3f getValue(const ColorSpace& s) const { Color c(value, cs); c.toColorSpace(s); return c.getValue(); }

		// get the color space
		const ColorSpace getColorSpace() const { return cs; }

		// computes the color distance of this color
		// to the specified color c. It checks that
		// the colors are both in the same color space.
		// If needed it transforms a copy of c in the
		// same color space as this color.
		float distanceFrom(const Color& c) const;

		// transforms this color in the color space
		// cs specified.
		void toColorSpace(const ColorSpace& cs);

		// transforms this color in the same color space
		// as the color c specified.
		void toColorSpaceOf(const Color& c) { toColorSpace(c.getColorSpace()); }

		// true if this color has color space s
		bool hasColorSpace(const ColorSpace& s) const { return cs == s; }

		// computes the color distance between color
		// c1 and color c2, both in the same color space.
		// If the color space differs, than transforms a
		// copy of c2 in the same color space as c1 first.
		static float distance(const Color& c1, const Color& c2) { return c1.distanceFrom(c2); }

		// computes the color distance between color
		// c1 and color c2, both in the color space cs.
		// If needed it locally transforms the color
		// in the wanted color space first.
		static float distance(const Color& c1, const Color& c2, const ColorSpace& cs);

		// compares two colors by their value norm based
		// on the RGB color space
		static bool greaterColor(const Color& c1, const Color& c2);

		// detection specific variables
		static int currentDetectionIndex; // used for converting RGB to detection confidence of current detection
		static int numDetectionIndices;
		static int detectionType; // used to switch between cleaned (2) and raw detection confidence (0)

	private:

		Eigen::Vector3f value; // the color value
		ColorSpace cs; // the color space relative to the color value


		// several color space transformations
		Eigen::Vector3f rgb2hcv(const Eigen::Vector3f& c);
		Eigen::Vector3f rgb2hsv(const Eigen::Vector3f& c);
		Eigen::Vector4f rgb2nci(const Eigen::Vector3f& c);
		Eigen::Vector3f rgb2cgz(const Eigen::Vector3f& c);
		Eigen::Vector3f rgb2cmd(const Eigen::Vector3f& c);
		Eigen::Vector3f rgb2rgc(const Eigen::Vector3f& c);
		Eigen::Vector3f hsv2rgb(const Eigen::Vector3f& c);
		Eigen::Vector3f hsv2hcv(const Eigen::Vector3f& c);
		Eigen::Vector3f hsv2cgz(const Eigen::Vector3f& c);
		Eigen::Vector3f cgz2rgb(const Eigen::Vector3f& c);
		Eigen::Vector3f cgz2hsv(const Eigen::Vector3f& c);
		Eigen::Vector3f cgz2hcv(const Eigen::Vector3f& c);
		Eigen::Vector3f hcv2rgb(const Eigen::Vector3f& c);
		Eigen::Vector3f hcv2hsv(const Eigen::Vector3f& c);
		Eigen::Vector3f hcv2cgz(const Eigen::Vector3f& c);
		Eigen::Vector3f cmd2rgb(const Eigen::Vector3f& c);
		Eigen::Vector3f rgc2rgb(const Eigen::Vector3f& c);
		Eigen::Vector3f cgt2rgb(const Eigen::Vector3f& c);
		Eigen::Vector3f rgb2cgt(const Eigen::Vector3f& c);
		float modulo(float x, float y);
};

//==============================================================================================//