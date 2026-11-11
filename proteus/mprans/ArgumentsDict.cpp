#include "pybind11/pybind11.h"
#include "pybind11/stl_bind.h"

#include "ArgumentsDict.h"

namespace py = pybind11;

namespace
{
    const std::string DARRAY_DICT_NAME = "DArrayDict";
    const std::string IARRAY_DICT_NAME = "IArrayDict";
    const std::string DSCALAR_DICT_NAME = "DScalarDict";
    const std::string ISCALAR_DICT_NAME = "IScalarDict";
    const std::string LARRAY_DICT_NAME = "LArrayDict";
    const std::string LSCALAR_DICT_NAME = "LScalarDict";
}

namespace proteus
{
    namespace detail
    {
        template <class OS, class M>
        OS& print_map(OS& os, const std::string& name, const M& m)
        {
            os << name << '{';
            bool f = false;
            for(const auto& kv: m)
            {
                if(f)
                {
                    os << ", ";
                }
                os << kv.first <<": " << kv.second;
                f = true;
            }
            os << '}';
            return os;
        }
    }

    template <class M>
    void bind_throwing_map(py::class_<M>& cl, const std::string& name)
    {
        using map_type = M;
        using key_type = typename map_type::key_type;
        using mapped_type = typename map_type::mapped_type;

        cl.def(py::init<>());

        cl.def("__setitem__",
                [](map_type& m, key_type& k, mapped_type& v)
                {
                    m.insert_or_assign(std::move(k), std::move(v));
                }
        );

        cl.def("__repr__",
                [name](map_type& m)
                {
                    std::ostringstream oss;
                    return detail::print_map(oss, name, m).str();
                },
               "Return the canonical representation of this map."
        );

        cl.def("__bool__",
                [](map_type& m) -> bool
                {
                    return !m.empty();
                },
                "Check wether the map is nonempty"
        );

        cl.def("__getitem__",
                [](map_type& m, const key_type& k) -> mapped_type&
                {
                    auto it = m.find(k);
                    if(it == m.end())
                    {
                        throw std::runtime_error(detail::key_error_message(k));
                    }
                    return it->second;
                }
        );
    }

    void bind_arguments_dict(py::class_<arguments_dict>& cl, const std::string& name)
    {
        cl.def(py::init<>());

        cl.def_readwrite("darray", &arguments_dict::m_darray);
        cl.def_readwrite("iarray", &arguments_dict::m_iarray);
        cl.def_readwrite("dscalar", &arguments_dict::m_dscalar);
        cl.def_readwrite("larray", &arguments_dict::m_larray);
        cl.def_readwrite("iscalar", &arguments_dict::m_iscalar);
        cl.def_readwrite("lscalar", &arguments_dict::m_lscalar);

        cl.def("__setitem__",
                [](arguments_dict& ad, std::string& k, proteus::pyarray<int>& a)
                {
                    ad.m_iarray.insert_or_assign(std::move(k), std::move(a));
                });
        cl.def("__setitem__",
                [](arguments_dict& ad, std::string& k, proteus::pyarray<std::int64_t>& a)
                {
                    ad.m_larray.insert_or_assign(std::move(k), std::move(a));
                });
        cl.def("__setitem__",
                [](arguments_dict& ad, std::string& k, proteus::pyarray<double>& a)
                {
                    ad.m_darray.insert_or_assign(std::move(k), std::move(a));
                });
        // IMPORTANT: keep the scalar overloads AFTER the pyarray overloads,
        // because numpy array containing a single elements can be implicitly
        // converted to scalars. If the scalar overload is define before the
        // pyarray overload, it is chosen by pybind11 and the array ends up
        // in the wrong dictionary.
        cl.def("__setitem__",
                [](arguments_dict& ad, std::string& k, std::int64_t i)
                {
                    if (i >= INT32_MIN && i <= INT32_MAX)
                    {
                        ad.m_iscalar.insert_or_assign(std::string(k), static_cast<int>(i));
                    }
                    ad.m_lscalar.insert_or_assign(std::move(k), std::move(i));
                });
        cl.def("__setitem__",
                [](arguments_dict& ad, std::string& k, double d)
                {
                    ad.m_dscalar.insert_or_assign(std::move(k), std::move(d));
                });
        cl.def("__repr__",
                [name](arguments_dict& ad)
                {
                    std::ostringstream oss;
                    oss << name << "{\n";
                    oss << "  " << "m_darray: ";
                    detail::print_map(oss, DARRAY_DICT_NAME, ad.m_darray);
                    oss << "\n  " << "m_iarray: ";
                    detail::print_map(oss, IARRAY_DICT_NAME, ad.m_iarray);
                    oss << "\n  " << "m_larray: ";
                    detail::print_map(oss, LARRAY_DICT_NAME, ad.m_larray);
                    oss << "\n  " << "m_dscalar: ";
                    detail::print_map(oss, DSCALAR_DICT_NAME, ad.m_dscalar);
                    oss << "\n  " << "m_iscalar: ";
                    detail::print_map(oss, ISCALAR_DICT_NAME, ad.m_iscalar);
                    oss << "\n  " << "m_lscalar: ";
                    detail::print_map(oss, LSCALAR_DICT_NAME, ad.m_lscalar);
                    oss << "\n}";
                    return oss.str();
                });
    }
}

PYBIND11_MODULE(cArgumentsDict, m)
{
    using proteus::pyarray_dict;
    using proteus::scalar_dict;
    using proteus::arguments_dict;

    proteus::import_numpy();

    using dpyarray_dict = pyarray_dict<std::string, double>;
    using ipyarray_dict = pyarray_dict<std::string, int>;
    using dscalar_dict = scalar_dict<std::string, double>;
    using iscalar_dict = scalar_dict<std::string, int>;
    using lpyarray_dict = pyarray_dict<std::string, std::int64_t>;
    using lscalar_dict = scalar_dict<std::string, std::int64_t>;

    py::class_<dpyarray_dict> dad(m, DARRAY_DICT_NAME.c_str());
    proteus::bind_throwing_map(dad, DARRAY_DICT_NAME.c_str());

    py::class_<ipyarray_dict> iad(m, IARRAY_DICT_NAME.c_str());
    proteus::bind_throwing_map(iad, IARRAY_DICT_NAME.c_str());

    py::class_<dscalar_dict> dsd(m, DSCALAR_DICT_NAME.c_str());
    proteus::bind_throwing_map(dsd, DSCALAR_DICT_NAME.c_str());

    py::class_<iscalar_dict> isd(m, ISCALAR_DICT_NAME.c_str());
    proteus::bind_throwing_map(isd, ISCALAR_DICT_NAME.c_str());

    py::class_<lpyarray_dict> lad(m, LARRAY_DICT_NAME.c_str());
    proteus::bind_throwing_map(lad, LARRAY_DICT_NAME.c_str());

    py::class_<lscalar_dict> lsd(m, LSCALAR_DICT_NAME.c_str());
    proteus::bind_throwing_map(lsd, LSCALAR_DICT_NAME.c_str());

    py::class_<arguments_dict> ad(m, "ArgumentsDict");
    proteus::bind_arguments_dict(ad, "ArgumentsDict");
}

