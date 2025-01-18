/* ---------------------------------------------------------------------
 * -- High Performance Computing Linpack Benchmark (HPL)
 *    HPL - 2.2 - February 24, 2016
 *    Antoine P. Petitet
 *    University of Tennessee, Knoxville
 *    Innovative Computing Laboratory
 *    (C) Copyright 2000-2008 All Rights Reserved
 *
 *    Modified by: Noel Chalmers
 *    (C) 2018-2022 Advanced Micro Devices, Inc.
 *    See the rocHPL/LICENCE file for details.
 *
 *    SPDX-License-Identifier: (BSD-3-Clause)
 * ---------------------------------------------------------------------
 */
#ifndef HPL_PAUXIL_HPP
#define HPL_PAUXIL_HPP
/*
 * ---------------------------------------------------------------------
 * Include files
 * ---------------------------------------------------------------------
 */
#include "hpl_misc.hpp"
#include "hpl_blas.hpp"
#include "hpl_auxil.hpp"

#include "hpl_pmisc.hpp"
#include "hpl_grid.hpp"

/*
 * ---------------------------------------------------------------------
 * #define macros definitions
 * ---------------------------------------------------------------------
 */
/*
 * Mindxg2p  returns the process coodinate owning the entry globally in-
 * dexed by ig_.
 */
#define Mindxg2p(ig_, inb_, nb_, proc_, src_, nprocs_)          \
  {                                                             \
    if(((ig_) >= (inb_)) && ((src_) >= 0) && ((nprocs_) > 1)) { \
      proc_ = (src_) + 1 + ((ig_) - (inb_)) / (nb_);            \
      proc_ -= (proc_ / (nprocs_)) * (nprocs_);                 \
    } else {                                                    \
      proc_ = (src_);                                           \
    }                                                           \
  }

#define Mindxg2l(il_, ig_, inb_, nb_, proc_, src_, nprocs_)               \
  {                                                                       \
    if(((ig_) < (inb_)) || ((src_) == -1) || ((nprocs_) == 1)) {          \
      il_ = (ig_);                                                        \
    } else {                                                              \
      int i__, j__;                                                       \
      j__ = (i__ = ((ig_) - (inb_)) / (nb_)) / (nprocs_);                 \
      il_ = (nb_) * (j__ - i__) +                                         \
            ((i__ + 1 - (j__ + 1) * (nprocs_)) ? (ig_) - (inb_) : (ig_)); \
    }                                                                     \
  }

#define Mindxg2lp(il_, proc_, ig_, inb_, nb_, src_, nprocs_)              \
  {                                                                       \
    if(((ig_) < (inb_)) || ((src_) == -1) || ((nprocs_) == 1)) {          \
      il_   = (ig_);                                                      \
      proc_ = (src_);                                                     \
    } else {                                                              \
      int i__, j__;                                                       \
      j__ = (i__ = ((ig_) - (inb_)) / (nb_)) / (nprocs_);                 \
      il_ = (nb_) * (j__ - i__) +                                         \
            ((i__ + 1 - (j__ + 1) * (nprocs_)) ? (ig_) - (inb_) : (ig_)); \
      proc_ = (src_) + 1 + i__;                                           \
      proc_ -= (proc_ / (nprocs_)) * (nprocs_);                           \
    }                                                                     \
  }
/*
 * Mindxl2g computes the global index ig_ corresponding to the local
 * index il_ in process proc_.
 */
#define Mindxl2g(ig_, il_, inb_, nb_, proc_, src_, nprocs_)                   \
  {                                                                           \
    if(((src_) >= 0) && ((nprocs_) > 1)) {                                    \
      if((proc_) == (src_)) {                                                 \
        if((il_) < (inb_))                                                    \
          ig_ = (il_);                                                        \
        else                                                                  \
          ig_ =                                                               \
              (il_) + (nb_) * ((nprocs_)-1) * (((il_) - (inb_)) / (nb_) + 1); \
      } else if((proc_) < (src_)) {                                           \
        ig_ = (il_) + (inb_) +                                                \
              (nb_) * (((nprocs_)-1) * ((il_) / (nb_)) + (proc_) - (src_)-1 + \
                       (nprocs_));                                            \
      } else {                                                                \
        ig_ = (il_) + (inb_) +                                                \
              (nb_) * (((nprocs_)-1) * ((il_) / (nb_)) + (proc_) - (src_)-1); \
      }                                                                       \
    } else {                                                                  \
      ig_ = (il_);                                                            \
    }                                                                         \
  }
/*
 * MnumrocI computes the # of local indexes  np_ residing in the process
 * of coordinate  proc_  corresponding to the interval of global indexes
 * i_:i_+n_-1  assuming  that the global index 0 resides in  the process
 * src_,  and that the indexes are distributed from src_ using the para-
 * meters inb_, nb_ and nprocs_.
 */
#define MnumrocI(np_, n_, i_, inb_, nb_, proc_, src_, nprocs_)              \
  {                                                                         \
    if(((src_) >= 0) && ((nprocs_) > 1)) {                                  \
      int inb__, mydist__, n__, nblk__, quot__, src__;                      \
      if((inb__ = (inb_) - (i_)) <= 0) {                                    \
        nblk__ = (-inb__) / (nb_) + 1;                                      \
        src__  = (src_) + nblk__;                                           \
        src__ -= (src__ / (nprocs_)) * (nprocs_);                           \
        inb__ += nblk__ * (nb_);                                            \
        if((n__ = (n_)-inb__) <= 0) {                                       \
          if((proc_) == src__)                                              \
            np_ = (n_);                                                     \
          else                                                              \
            np_ = 0;                                                        \
        } else {                                                            \
          if((mydist__ = (proc_)-src__) < 0) mydist__ += (nprocs_);         \
          nblk__ = n__ / (nb_) + 1;                                         \
          mydist__ -= nblk__ - (quot__ = (nblk__ / (nprocs_))) * (nprocs_); \
          if(mydist__ < 0) {                                                \
            if((proc_) != src__)                                            \
              np_ = (nb_) + (nb_)*quot__;                                   \
            else                                                            \
              np_ = inb__ + (nb_)*quot__;                                   \
          } else if(mydist__ > 0) {                                         \
            np_ = (nb_)*quot__;                                             \
          } else {                                                          \
            if((proc_) != src__)                                            \
              np_ = n__ + (nb_) + (nb_) * (quot__ - nblk__);                \
            else                                                            \
              np_ = (n_) + (nb_) * (quot__ - nblk__);                       \
          }                                                                 \
        }                                                                   \
      } else {                                                              \
        if((n__ = (n_)-inb__) <= 0) {                                       \
          if((proc_) == (src_))                                             \
            np_ = (n_);                                                     \
          else                                                              \
            np_ = 0;                                                        \
        } else {                                                            \
          if((mydist__ = (proc_) - (src_)) < 0) mydist__ += (nprocs_);      \
          nblk__ = n__ / (nb_) + 1;                                         \
          mydist__ -= nblk__ - (quot__ = (nblk__ / (nprocs_))) * (nprocs_); \
          if(mydist__ < 0) {                                                \
            if((proc_) != (src_))                                           \
              np_ = (nb_) + (nb_)*quot__;                                   \
            else                                                            \
              np_ = inb__ + (nb_)*quot__;                                   \
          } else if(mydist__ > 0) {                                         \
            np_ = (nb_)*quot__;                                             \
          } else {                                                          \
            if((proc_) != (src_))                                           \
              np_ = n__ + (nb_) + (nb_) * (quot__ - nblk__);                \
            else                                                            \
              np_ = (n_) + (nb_) * (quot__ - nblk__);                       \
          }                                                                 \
        }                                                                   \
      }                                                                     \
    } else {                                                                \
      np_ = (n_);                                                           \
    }                                                                       \
  }

#define Mnumroc(np_, n_, inb_, nb_, proc_, src_, nprocs_) \
  MnumrocI(np_, n_, 0, inb_, nb_, proc_, src_, nprocs_)
/*
 * ---------------------------------------------------------------------
 * Function prototypes
 * ---------------------------------------------------------------------
 */
void HPL_indxg2lp(int*,
                  int*,
                  const int,
                  const int,
                  const int,
                  const int,
                  const int);

int HPL_indxg2l(const int, const int, const int, const int, const int);
int HPL_indxg2p(const int, const int, const int, const int, const int);

int HPL_indxl2g(const int,
                const int,
                const int,
                const int,
                const int,
                const int);

void HPL_infog2l(int,
                 int,
                 const int,
                 const int,
                 const int,
                 const int,
                 const int,
                 const int,
                 const int,
                 const int,
                 const int,
                 const int,
                 int*,
                 int*,
                 int*,
                 int*);

int HPL_numroc(const int,
               const int,
               const int,
               const int,
               const int,
               const int);

int HPL_numrocI(const int,
                const int,
                const int,
                const int,
                const int,
                const int,
                const int);

void HPL_dlaswp00N(const int, const int, double*, const int, const int*);

void HPL_dlaswp01T(const int,
                   const int,
                   double*,
                   const int,
                   double*,
                   const int,
                   const int*);

void HPL_dlaswp02T(const int,
                   const int,
                   double*,
                   const int,
                   const int*,
                   const int*);

void HPL_dlaswp03T(const int,
                   const int,
                   double*,
                   const int,
                   double*,
                   const int,
                   const int*);

void HPL_dlaswp04T(const int,
                   const int,
                   double*,
                   const int,
                   double*,
                   const int,
                   const int*);

void HPL_dlaswp10N(const int, const int, double*, const int, const int*);

void HPL_pabort(int, const char*, const char*, ...);
void HPL_pwarn(FILE*, int, const char*, const char*, ...);

void HPL_pdlaprnt(const HPL_T_grid*,
                  const int,
                  const int,
                  const int,
                  double*,
                  const int,
                  const int,
                  const int,
                  const char*);

double HPL_pdlamch(MPI_Comm, const HPL_T_MACH);

double HPL_pdlange(const HPL_T_grid*,
                   const HPL_T_NORM,
                   const int,
                   const int,
                   const int,
                   const double*,
                   const int,
                         double*);

#endif
/*
 * End of hpl_pauxil.hpp
 */
