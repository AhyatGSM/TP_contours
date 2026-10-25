#ifndef EVALUATION_H
#define EVALUATION_H

#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

// evaluation.h
// Comparaison d'image de contours calculée avec l'image de référence
// contours tracés à la main

struct Mesures {

    int detectes;       // contours_détectés
    int reference;      // contours_référence
    int corrects;       // contours_corrects
    int fauxPositifs;   // faux_positifs
    int fauxNegatifs;   // faux_négatifs

    double P;           // performance
    double TFP;         // taux de faux positifs
    double TFN;         // taux de faux négatifs

    double precision;   // part des pixels détectés qui sont justes
    double rappel;      // part des contours de référence retrouvés
    double fMesure;     // compromis entre précision et rappel
};

// Référence au même format que nos résultats : contours en BLANC 
cv::Mat preparerReference(const cv::Mat& gt)
{
    cv::Mat bin;
    cv::threshold(gt, bin, 127, 255, cv::THRESH_BINARY);

    // Si l'image est majoritairement blanche, les contours sont noirs : on inverse.
    if (cv::countNonZero(bin) > bin.rows * bin.cols / 2)
        bin = 255 - bin;

    return bin;
}

// Zone de tolérance autour des contours 
// Avec voisinage : contours "grossis" d'un pixel par dilatation 3x3.
// Sans voisinage : les contours tels quels.
cv::Mat zoneTolerance(const cv::Mat& contours, bool voisinage)
{
    if (!voisinage) 
        return contours.clone(); //renvoie contour tels quels

    cv::Mat zone;
    cv::Mat element(3, 3, CV_8U, cv::Scalar(1));   // élément structurant 3x3
    cv::dilate(contours, zone, element);
    return zone;
}

// Calcul des mesures 
Mesures evaluer(const cv::Mat& detecte, const cv::Mat& reference, bool voisinage)
{
    Mesures m = {0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    m.detectes = cv::countNonZero(detecte);
    m.reference = cv::countNonZero(reference);

    cv::Mat zoneReference = zoneTolerance(reference, voisinage);
    cv::Mat zoneDetecte = zoneTolerance(detecte, voisinage);

    // Pixels détectés qui tombent sur un contour de référence (ou à côté)
    m.corrects = cv::countNonZero(detecte & zoneReference);
    m.fauxPositifs = m.detectes - m.corrects;

    // Pixels de référence qui ont un pixel détecté sur eux (ou à côté) ;
    // les autres sont des contours manqués = faux négatifs
    int referenceTrouves = cv::countNonZero(reference & zoneDetecte);
    m.fauxNegatifs = m.reference - referenceTrouves;

    // Les 3 mesures du sujet
    int total = m.corrects + m.fauxPositifs + m.fauxNegatifs;
    if (total > 0) {
        m.P = (double)m.corrects / total;       //performance
        m.TFP = (double)m.fauxPositifs / total; //taux faux positif
        m.TFN = (double)m.fauxNegatifs / total; //taux faux negatif
    }

    // Critères supplémentaires
    if (m.detectes > 0)
        m.precision = (double)m.corrects / m.detectes;
    if (m.reference > 0)
        m.rappel = (double)referenceTrouves / m.reference;
    if (m.precision + m.rappel > 0)
        m.fMesure = 2 * m.precision * m.rappel / (m.precision + m.rappel);

    return m;
}

// Image des erreurs en couleur, vert = contour correct, rouge = faux positif, bleu = faux négatif
cv::Mat imageErreurs(const cv::Mat& detecte, const cv::Mat& reference, bool voisinage)
{
    cv::Mat zoneReference = zoneTolerance(reference, voisinage);
    cv::Mat zoneDetecte = zoneTolerance(detecte, voisinage);

    cv::Mat corrects = detecte & zoneReference;        // détecté et attendu (vert)
    cv::Mat fauxPositifs = detecte & ~zoneReference;   // détecté mais pas attendu (rouge) 
    cv::Mat fauxNegatifs = reference & ~zoneDetecte;   // attendu mais pas détecté (bleu)

    std::vector<cv::Mat> canaux = {fauxNegatifs, corrects, fauxPositifs};
    cv::Mat couleur;
    cv::merge(canaux, couleur);
    return couleur;
}

#endif