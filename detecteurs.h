#ifndef DETECTEURS_H
#define DETECTEURS_H

#include <algorithm>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "edgedetector.h"  
#include "laplacianZC.h"    


// En Entrée : image en niveaux de gris.
// Sortie : image binaire, contours en blanc (255) sur fond noir (0).

// Taille des noyaux de Sobel et du Laplacien (3x3)
const int TAILLE_NOYAU = 3;

// Flou gaussien (chapitre 6, filters.cpp)
// Réduit le bruit avant de dériver. sigma = 0 : pas de flou
cv::Mat lisser(const cv::Mat& image, double sigma)
{
    cv::Mat resultat;
    if (sigma > 0)
        cv::GaussianBlur(image, resultat, cv::Size(0, 0), sigma);  // taille déduite de sigma
    else
        resultat = image.clone();
    return resultat;
}

// Sobel : dérivée première 
// Contour = pixel où la norme du gradient dépasse le seuil
cv::Mat detecteurSobel(const cv::Mat& image, double sigma, int seuil)
{
    cv::Mat lisse = lisser(image, sigma);

    // La classe du cours calcule gx, gy, puis la norme racine(gx² + gy²)
    EdgeDetector ed;
    ed.setAperture(TAILLE_NOYAU);
    ed.computeSobel(lisse);

    // Seuillage de la norme : 255 si norme > seuil, 0 sinon.
    cv::Mat contours;
    cv::threshold(ed.getMagnitude(), contours, seuil, 255, cv::THRESH_BINARY); //ici on seuil la norme
    contours.convertTo(contours, CV_8U);   // float -> 8 bits
    return contours;
}

// Laplace : dérivée seconde  
// Contour = passage par zéro du Laplacien, à condition qu'il soit "fort"
cv::Mat detecteurLaplace(const cv::Mat& image, double sigma, int seuil)
{
    // Le flou est indispensable : le Laplacien est très sensible au bruit
    cv::Mat lisse = lisser(image, sigma);

    // Laplacien et passages par zéro avec la classe du chapitre 6
    LaplacianZC laplacien;
    laplacien.setAperture(TAILLE_NOYAU);
    cv::Mat lap = laplacien.computeLaplacian(lisse);     // valeurs positives et négatives
    cv::Mat passages = laplacien.getZeroCrossings(lap);  // 255 là où le signe change

    // Force du passage : écart entre le max et le min du Laplacien autour du pixel
    // (gradient morphologique vu au chapitre 5)
    cv::Mat pente;
    cv::morphologyEx(lap, pente, cv::MORPH_GRADIENT, cv::Mat());

    // On garde les passages dont la force dépasse le seuil
    cv::Mat fort;
    cv::threshold(pente, fort, seuil, 255, cv::THRESH_BINARY);
    fort.convertTo(fort, CV_8U);

    cv::Mat contours = passages & fort;   // passage par zéro ET assez fort
    return contours;
}

// Canny  
// Flou + gradient + suppression des non-maxima + seuillage par hystérésis
cv::Mat detecteurCanny(const cv::Mat& image, double sigma, int seuilBas, int seuilHaut)
{
    cv::Mat lisse = lisser(image, sigma);

    cv::Mat contours;
    cv::Canny(lisse, contours,
              seuilBas,       // seuil bas de l'hystérésis
              seuilHaut,      // seuil haut de l'hystérésis
              TAILLE_NOYAU,   // noyau de Sobel utilisé en interne
              true);          // norme racine(gx² + gy²), comme notre Sobel
    return contours;
}

// Choix du détecteur 
// detecteur : 0 = Sobel, 1 = Laplace, 2 = Canny
// seuil : seuil de Sobel et Laplace, et seuil BAS de Canny
// seuilHaut : seuil haut de Canny (ignoré par les deux autres)
cv::Mat detecter(const cv::Mat& image, int detecteur, double sigma, int seuil, int seuilHaut)
{
    if (detecteur == 0)
        return detecteurSobel(image, sigma, seuil);
    if (detecteur == 1)
        return detecteurLaplace(image, sigma, seuil);
    return detecteurCanny(image, sigma, seuil, seuilHaut);
}

#endif