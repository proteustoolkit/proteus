#ifndef ARGUMENTSDICT_H
#define ARGUMENTSDICT_H

#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include "pyarray.h"

namespace proteus
{

    /****************
     * throwing_map *
     ****************/

    // Special map that behaves like a standard map except for
    // - operator[] which throws instead of inserting a default
    //   value when a key is not found
    // - insert and insert_or_assign which always move the key
    //   and the value. This allows to keep pyarrays synchronized
    //   with the numpy arrays they refer to, while avoiding code
    //   duplication for array of scalars (because insert_or_assign
    //   is not available in C++14)
    template <class K, class T>
    class throwing_map : private std::map<K, T>
    {
    public:

        using base_type = std::map<K, T>;
        using key_type = typename base_type::key_type;
        using mapped_type = typename base_type::mapped_type;
        using value_type = typename base_type::value_type;
        using reference = typename base_type::reference;
        using const_reference = typename base_type::const_reference;
        using pointer = typename base_type::pointer;
        using const_pointer = typename base_type::const_pointer;
        using size_type = typename base_type::size_type;
        using difference_type = typename base_type::difference_type;
        using iterator = typename base_type::iterator;
        using const_iterator = typename base_type::const_iterator;

        using base_type::empty;
        using base_type::find;
        using base_type::emplace;

        using base_type::begin;
        using base_type::end;
        using base_type::cbegin;
        using base_type::cend;

        // Throws if k is not in the map
        mapped_type& operator[](const key_type& k);
    
        std::pair<iterator, bool> insert(value_type&&);
        std::pair<iterator, bool> insert_or_assign(key_type&& k, mapped_type&& v);
    };

    /******************
     * arguments_dict *
     ******************/

    template <class K, class T>
    using pyarray_dict = throwing_map<K, proteus::pyarray<T>>;

    template <class K, class T>
    using scalar_dict = throwing_map<K, T>;

    struct arguments_dict
    {
        pyarray_dict<std::string, double> m_darray;
        pyarray_dict<std::string, int> m_iarray;
        // 64-bit integer arrays: the mesh and CSR index arrays when
        // proteus_index_t / proteus_csr_index_t are 64 bits (index_type.h).
        // An array lands in the dict of its own dtype, nothing is converted,
        // so a kernel that asks for the wrong width gets a key error that
        // names the dict the array is in.
        pyarray_dict<std::string, std::int64_t> m_larray;
        scalar_dict<std::string, double> m_dscalar;
        scalar_dict<std::string, int> m_iscalar;
        // A Python int is stored in m_lscalar always, and also in m_iscalar
        // when it fits an int; a count that does not fit is only in m_lscalar.
        scalar_dict<std::string, std::int64_t> m_lscalar;

        template <class T>
        proteus::pyarray<T>& array(const std::string& key);

        template <class T>
        T& scalar(const std::string& key);

    private:

        template <class D1>
        typename D1::mapped_type& find_element(const std::string& key,
                                               D1& expected_dict,
                                               const std::string& expected_type);

        std::string where_is(const std::string& key) const;
    };

    /*******************************
     * throwing_map implementation *
     *******************************/

    namespace detail
    {
        template <class T>
        inline std::string key_error_messagei(const T& key)
        {
            return key_error_message(std::to_string(key));
        }

        inline std::string key_error_message(const std::string& key)
        {
            return std::string("key ") + key + std::string(" not found");
        }
    }

    template <class K, class T>
    inline auto throwing_map<K, T>::operator[](const key_type& k) -> mapped_type&
    {
        auto it = this->find(k);
        if(it == end())
        {
            throw std::runtime_error(detail::key_error_message(k));
        }
        return it->second;
    }

    template <class K, class T>
    inline auto throwing_map<K, T>::insert(value_type&& v) -> std::pair<iterator, bool>
    {
        return base_type::insert(std::move(v));
    }

    template <class K, class T>
    inline auto throwing_map<K, T>::insert_or_assign(key_type&& k, mapped_type&& v) -> std::pair<iterator, bool>
    {
        auto it = base_type::find(k);
        if(it == base_type::end())
        {
            return base_type::emplace(k, std::move(v));
        }
        else
        {
            it->second = std::move(v);
            return std::make_pair(it, false);
        }
    }

    /*********************************
     * arguments_dict implementation *
     *********************************/

    template <>
    inline proteus::pyarray<double>& arguments_dict::array<double>(const std::string& key)
    {
        return find_element(key, m_darray, "pyarray<double>");
    }

    template <>
    inline proteus::pyarray<int>& arguments_dict::array<int>(const std::string& key)
    {
        return find_element(key, m_iarray, "pyarray<int>");
    }

    template <>
    inline proteus::pyarray<std::int64_t>& arguments_dict::array<std::int64_t>(const std::string& key)
    {
        return find_element(key, m_larray, "pyarray<int64>");
    }

    template <>
    inline double& arguments_dict::scalar<double>(const std::string& key)
    {
        return find_element(key, m_dscalar, "double scalar");
    }

    template <>
    inline int& arguments_dict::scalar<int>(const std::string& key)
    {
        return find_element(key, m_iscalar, "int scalar");
    }

    template <>
    inline std::int64_t& arguments_dict::scalar<std::int64_t>(const std::string& key)
    {
        return find_element(key, m_lscalar, "int64 scalar");
    }

    template <class D1>
    typename D1::mapped_type& arguments_dict::find_element(const std::string& key,
                                                           D1& expected_dict,
                                                           const std::string& expected_type)
    {
        try
        {
            return expected_dict[key];
        }
        catch(std::runtime_error& e)
        {
            throw std::runtime_error(e.what() + std::string(" in dict of ") + expected_type + where_is(key));
        }
    }

    inline std::string arguments_dict::where_is(const std::string& key) const
    {
        std::string found;
        auto note = [&](const auto& d, const char* name)
        {
            if (d.find(key) != d.cend())
            {
                found += std::string(found.empty() ? "" : ", ") + name;
            }
        };
        note(m_darray, "pyarray<double>");
        note(m_iarray, "pyarray<int>");
        note(m_larray, "pyarray<int64>");
        note(m_dscalar, "double scalar");
        note(m_iscalar, "int scalar");
        note(m_lscalar, "int64 scalar");
        if (found.empty())
        {
            return " or in any of the internal dicts";
        }
        return " but found in dict of " + found;
    }
}

#endif

