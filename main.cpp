#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "detecteurs.h"
#include "evaluation.h"


// TP Détection de contours : comparaison de Sobel, Laplace et Canny

// Utilisation (les touches se tapent dans la fenêtre d'images, pas dans le terminal) :
//  - curseurs : choix de l'image, du détecteur et de ses paramètres
//  - l : applique les réglages des curseurs à toutes les images
//  - a : série automatique (3 réglages par détecteur, avec et sans voisinage)
//  - s : screeshot de l'affichage actuel (rapport)
//  - q : quitter

// Dossier des images, par rapport au dossier build d'où on lance le programme
const std::string DOSSIER = "../contours/images";

const std::string FENETRE = "TP contours";
const std::string NOMS[3] = {"Sobel", "Laplace", "Canny"};

//  Un réglage = un détecteur et ses paramètres 
struct Reglage {
    int detecteur;   // 0 Sobel, 1 Laplace, 2 Canny
    int flou;        // sigma x 10 (10 -> sigma = 1,0)
    int seuil;       // seuil (seuil bas pour Canny)
    int seuilHaut;   // seuil haut (Canny seulement)
};

// Réglages testés par la série automatique (touche a)
// (detecteur, flou, seuil, seuilHaut)
const std::vector<Reglage> SERIE_AUTO = {
    {0, 14,  55,   0},   // Sobel   : seuil faible
    {0, 14,  92,   0},   //           seuil moyen  (le + favorable selon mes tests)
    {0, 14, 138,   0},   //           seuil fort
    {1, 19,  13,   0},   // Laplace : seuil faible
    {1, 19,  21,   0},   //           seuil moyen  (le + favorable selon mes tests)
    {1, 19,  32,   0},   //           seuil fort
    {2, 14,  32,  51},   // Canny   : seuils faibles
    {2, 14,  54,  85},   //           seuils moyens (le + favorable selon mes tests)
    {2, 14,  81, 128},   //           seuils forts
};

//  Données chargées une seule fois au démarrage 
std::vector<std::string> noms;     // "basket", "bear" etc
std::vector<cv::Mat> images;       // images en niveaux de gris
std::vector<cv::Mat> references;   // références (contours en blanc)

// État de l'interface 
bool pret = false;          // true quand tous les curseurs sont créés
cv::Mat affichageComplet;   // dernier affichage en taille réelle (pour 's')
std::string nomCapture;     // nom du fichier pour 's'


// PETITS OUTILS pour l'affichage

// converti "../contours/images/basket.pgm" -> "basket"
std::string nomDepuisChemin(const std::string& chemin)
{
    size_t debut = chemin.find_last_of('/') + 1;   // après le dernier '/'
    size_t fin = chemin.find_last_of('.');          // avant le '.pgm'
    return chemin.substr(debut, fin - debut);
}

// Nombre entre 0 et 1 avec 3 décimales et une virgule : 0.423156 -> "0,423"
std::string nombre(double x)
{
    std::string s = std::to_string(x).substr(0, 5);
    s[1] = ',';   // remplace le point par une virgule
    return s;
}

// Flou x10 -> texte : 15 -> "1,5"
std::string texteSigma(int flou)
{
    return std::to_string(flou / 10) + "," + std::to_string(flou % 10);
}

// Description d'un réglage : "Canny  sigma=1,0  seuils=60/150"
std::string texteReglage(const Reglage& r)
{
    std::string texte = NOMS[r.detecteur] + "  sigma=" + texteSigma(r.flou);
    if (r.detecteur == 2)
        texte += "  seuils=" + std::to_string(r.seuil) + "/" + std::to_string(r.seuilHaut);
    else
        texte += "  seuil=" + std::to_string(r.seuil);
    return texte;
}

// Ajoute un titre au-dessus d'une image couleur (bandeau noir de 30 pixels)
cv::Mat avecTitre(const cv::Mat& image, const std::string& titre)
{
    cv::Mat resultat;
    cv::copyMakeBorder(image, resultat, 30, 0, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    cv::putText(resultat, titre, cv::Point(8, 22),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
    return resultat;
}

// =====================================================================
// CHARGEMENT DES IMAGES
// =====================================================================

// Charge toutes les images du dossier et leurs références
bool chargerDonnees()
{
    std::vector<cv::String> fichiers;
    cv::glob(DOSSIER + "/*.pgm", fichiers, false);   // false : ne regarde pas dans gt/

    for (int i = 0; i < (int)fichiers.size(); i++) {
        std::string nom = nomDepuisChemin(fichiers[i]);
        cv::Mat image = cv::imread(fichiers[i], cv::IMREAD_GRAYSCALE);
        cv::Mat gt = cv::imread(DOSSIER + "/gt/" + nom + "_gt_binary.pgm", cv::IMREAD_GRAYSCALE);

        if (image.empty() || gt.empty()) {
            std::cerr << "Attention : " << nom << " ou sa reference introuvable, image ignoree" << std::endl;
            continue;
        }

        // Sécurité : la référence doit avoir la même taille que l'image
        if (gt.size() != image.size())
            cv::resize(gt, gt, image.size(), 0, 0, cv::INTER_NEAREST);

        noms.push_back(nom);
        images.push_back(image);
        references.push_back(preparerReference(gt));
    }

    std::cout << images.size() << " images chargees depuis " << DOSSIER << std::endl;
    return !images.empty();
}

// =====================================================================
// INTERFACE : CURSEURS ET AFFICHAGE
// =====================================================================

// Lit le réglage choisi avec les curseurs
Reglage reglageCurseurs()
{
    Reglage r;
    r.detecteur = cv::getTrackbarPos("Detecteur", FENETRE);
    r.flou = cv::getTrackbarPos("Flou x10", FENETRE);
    r.seuil = cv::getTrackbarPos("Seuil", FENETRE);
    r.seuilHaut = cv::getTrackbarPos("Seuil haut", FENETRE);
    return r;
}

// Appelée à chaque mouvement d'un curseur : recalcule et réaffiche
void miseAJour(int, void*)
{
    if (!pret)
        return;

    // 1. Lecture des curseurs
    int numero = cv::getTrackbarPos("Image", FENETRE);
    Reglage r = reglageCurseurs();
    bool voisinage = (cv::getTrackbarPos("Voisinage", FENETRE) == 1);

    //  Détection et évaluation
    cv::Mat contours = detecter(images[numero], r.detecteur, r.flou / 10.0, r.seuil, r.seuilHaut);
    Mesures m = evaluer(contours, references[numero], voisinage);
    cv::Mat erreurs = imageErreurs(contours, references[numero], voisinage);

    // Les 4 vues en couleur (pour pouvoir les assembler avec l'image des erreurs)
    cv::Mat vueImage, vueReference, vueContours;
    cv::cvtColor(images[numero], vueImage, cv::COLOR_GRAY2BGR);
    cv::cvtColor(references[numero], vueReference, cv::COLOR_GRAY2BGR);
    cv::cvtColor(contours, vueContours, cv::COLOR_GRAY2BGR);

    // 4. Assemblage en grille 2 x 2
    cv::Mat haut, bas, grille;
    cv::hconcat(avecTitre(vueImage, "Image : " + noms[numero]),
                avecTitre(vueReference, "Reference (tracee a la main)"), haut);
    cv::hconcat(avecTitre(vueContours, "Contours detectes"),
                avecTitre(erreurs, "Vert = correct  Rouge = faux pos.  Bleu = faux neg."), bas);
    cv::vconcat(haut, bas, grille);

    // 5. Bandeau des résultats en haut
    cv::copyMakeBorder(grille, affichageComplet, 100, 0, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    std::string ligne1 = texteReglage(r) + "     voisinage 3x3 : " + (voisinage ? "oui" : "non");
    std::string ligne2 = "P = " + nombre(m.P) + "     TFP = " + nombre(m.TFP)
                       + "     TFN = " + nombre(m.TFN);
    std::string ligne3 = "Precision = " + nombre(m.precision) + "     Rappel = " + nombre(m.rappel)
                       + "     F-mesure = " + nombre(m.fMesure);

    cv::putText(affichageComplet, ligne1, cv::Point(10, 28),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
    cv::putText(affichageComplet, ligne2, cv::Point(10, 60),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
    cv::putText(affichageComplet, ligne3, cv::Point(10, 90),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);

    nomCapture = "capture_" + noms[numero] + "_" + NOMS[r.detecteur] + ".png";

    // 6. Affichage réduit pour tenir à l'écran
    cv::Mat reduit;
    cv::resize(affichageComplet, reduit, cv::Size(), 0.6, 0.6, cv::INTER_AREA);
    cv::imshow(FENETRE, reduit);
}

// =====================================================================
// TRAITEMENT D'UN LOT D'IMAGES
// =====================================================================

// Ligne de titres des fichiers CSV
void ecrireEntete(std::ofstream& csv)
{
    csv << "detecteur;sigma;seuil;seuil_haut;voisinage;image;"
        << "contours_detectes;contours_reference;contours_corrects;faux_positifs;faux_negatifs;"
        << "P;TFP;TFN;precision;rappel;f_mesure\n";
}

// Une ligne de résultats dans un fichier CSV
void ecrireLigne(std::ofstream& csv, const Reglage& r, bool voisinage,
                 const std::string& image, const Mesures& m)
{
    csv << NOMS[r.detecteur] << ";" << texteSigma(r.flou) << ";" << r.seuil << ";"
        << r.seuilHaut << ";" << (voisinage ? "oui" : "non") << ";" << image << ";"
        << m.detectes << ";" << m.reference << ";" << m.corrects << ";"
        << m.fauxPositifs << ";" << m.fauxNegatifs << ";"
        << nombre(m.P) << ";" << nombre(m.TFP) << ";" << nombre(m.TFN) << ";"
        << nombre(m.precision) << ";" << nombre(m.rappel) << ";" << nombre(m.fMesure) << "\n";
}

// Applique un réglage à toutes les images.
// Écrit une ligne par image dans le CSV, puis une ligne MOYENNE, et renvoie la moyenne.
Mesures traiterLot(const Reglage& r, bool voisinage, std::ofstream& csv)
{
    Mesures somme = {0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    int n = (int)images.size();

    for (int i = 0; i < n; i++) {
        cv::Mat contours = detecter(images[i], r.detecteur, r.flou / 10.0, r.seuil, r.seuilHaut);
        Mesures m = evaluer(contours, references[i], voisinage);
        ecrireLigne(csv, r, voisinage, noms[i], m);

        somme.detectes += m.detectes;
        somme.reference += m.reference;
        somme.corrects += m.corrects;
        somme.fauxPositifs += m.fauxPositifs;
        somme.fauxNegatifs += m.fauxNegatifs;
        somme.P += m.P;
        somme.TFP += m.TFP;
        somme.TFN += m.TFN;
        somme.precision += m.precision;
        somme.rappel += m.rappel;
        somme.fMesure += m.fMesure;
    }

    // Moyenne sur toutes les images
    Mesures moyenne = somme;
    moyenne.detectes /= n;
    moyenne.reference /= n;
    moyenne.corrects /= n;
    moyenne.fauxPositifs /= n;
    moyenne.fauxNegatifs /= n;
    moyenne.P /= n;
    moyenne.TFP /= n;
    moyenne.TFN /= n;
    moyenne.precision /= n;
    moyenne.rappel /= n;
    moyenne.fMesure /= n;

    ecrireLigne(csv, r, voisinage, "MOYENNE", moyenne);
    return moyenne;
}

// Affiche une moyenne dans le terminal
void afficherMoyenne(const Reglage& r, bool voisinage, const Mesures& m)
{
    std::cout << texteReglage(r) << "  voisinage " << (voisinage ? "oui" : "non")
              << "  ->  P = " << nombre(m.P) << "  TFP = " << nombre(m.TFP)
              << "  TFN = " << nombre(m.TFN) << "  F = " << nombre(m.fMesure) << std::endl;
}

// Touche l : lot avec les réglages actuels des curseurs
void lancerLotCurseurs()
{
    Reglage r = reglageCurseurs();
    bool voisinage = (cv::getTrackbarPos("Voisinage", FENETRE) == 1);

    // Le fichier existe-t-il déjà ? (sinon il faut écrire la ligne de titres)
    std::ifstream test("resultats_lot.csv");
    bool nouveauFichier = !test.good();
    test.close();

    // Mode "ajout" : les résultats des lots précédents sont conservés
    std::ofstream csv("resultats_lot.csv", std::ios::app);
    if (nouveauFichier)
        ecrireEntete(csv);

    std::cout << "\nLot sur " << images.size() << " images..." << std::endl;
    Mesures moyenne = traiterLot(r, voisinage, csv);
    afficherMoyenne(r, voisinage, moyenne);
    std::cout << "Resultats ajoutes dans resultats_lot.csv" << std::endl;
}

// Touche a : série automatique de tous les réglages de SERIE_AUTO,
void lancerSerieAuto()
{
    std::ofstream csv("resultats_auto.csv");           // détail image par image
    std::ofstream tableau("tableau_moyennes.csv");     // une ligne par réglage
    ecrireEntete(csv);
    tableau << "detecteur;sigma;seuil;seuil_haut;voisinage;P;TFP;TFN;precision;rappel;f_mesure\n";

    std::cout << "\nSerie automatique : " << SERIE_AUTO.size() * 2 << " lots, patientez..." << std::endl;

    for (int v = 1; v >= 0; v--) {          // d'abord avec voisinage, puis sans
        bool voisinage = (v == 1);

        for (int k = 0; k < (int)SERIE_AUTO.size(); k++) {
            Reglage r = SERIE_AUTO[k];
            Mesures moyenne = traiterLot(r, voisinage, csv);

            tableau << NOMS[r.detecteur] << ";" << texteSigma(r.flou) << ";" << r.seuil << ";"
                    << r.seuilHaut << ";" << (voisinage ? "oui" : "non") << ";"
                    << nombre(moyenne.P) << ";" << nombre(moyenne.TFP) << ";" << nombre(moyenne.TFN) << ";"
                    << nombre(moyenne.precision) << ";" << nombre(moyenne.rappel) << ";"
                    << nombre(moyenne.fMesure) << "\n";

            afficherMoyenne(r, voisinage, moyenne);
        }
    }

    std::cout << "Termine : tableau_moyennes.csv (tableau du rapport)"
              << " et resultats_auto.csv (detail image par image)" << std::endl;
}

// =====================================================================
// PROGRAMME PRINCIPAL
// =====================================================================

int main()
{
    // ===== 1. Chargement de toutes les images =====
    if (!chargerDonnees()) {
        std::cerr << "Erreur : aucune image chargee. Verifiez le dossier " << DOSSIER << std::endl;
        return -1;
    }

    // ===== 2. Fenêtre et curseurs =====
    cv::namedWindow(FENETRE, cv::WINDOW_AUTOSIZE);

    // createTrackbar(nom, fenêtre, nullptr, valeur max, fonction appelée)
    cv::createTrackbar("Image", FENETRE, nullptr, (int)images.size() - 1, miseAJour);
    cv::createTrackbar("Detecteur", FENETRE, nullptr, 2, miseAJour);     // 0 Sobel, 1 Laplace, 2 Canny
    cv::createTrackbar("Flou x10", FENETRE, nullptr, 50, miseAJour);     // sigma de 0 à 5
    cv::createTrackbar("Seuil", FENETRE, nullptr, 500, miseAJour);       // seuil bas pour Canny
    cv::createTrackbar("Seuil haut", FENETRE, nullptr, 500, miseAJour);  // Canny seulement
    cv::createTrackbar("Voisinage", FENETRE, nullptr, 1, miseAJour);     // 1 : tolérance 3x3

    // Positions de départ
    cv::setTrackbarPos("Detecteur", FENETRE, 2);
    cv::setTrackbarPos("Flou x10", FENETRE, 10);
    cv::setTrackbarPos("Seuil", FENETRE, 60);
    cv::setTrackbarPos("Seuil haut", FENETRE, 150);
    cv::setTrackbarPos("Voisinage", FENETRE, 1);

    pret = true;
    miseAJour(0, nullptr);   // premier affichage

    std::cout << "\nDetecteur : 0 = Sobel, 1 = Laplace, 2 = Canny" << std::endl;
    std::cout << "Touches (cliquer d'abord sur la fenetre d'images) :" << std::endl;
    std::cout << "  l : lot avec les reglages actuels -> resultats_lot.csv" << std::endl;
    std::cout << "  a : serie automatique -> tableau_moyennes.csv" << std::endl;
    std::cout << "  s : enregistrer l'affichage actuel" << std::endl;
    std::cout << "  q : quitter" << std::endl;

    // ===== 3. Boucle des touches =====
    while (true) {
        int touche = cv::waitKey(0);

        if (touche == 'q' || touche == 27 || touche == -1)   // 27 = Echap, -1 = fenêtre fermée
            break;

        if (touche == 'l')
            lancerLotCurseurs();

        if (touche == 'a')
            lancerSerieAuto();

        if (touche == 's') {
            cv::imwrite(nomCapture, affichageComplet);
            std::cout << "Image enregistree : " << nomCapture << std::endl;
        }
    }

    return 0;
}