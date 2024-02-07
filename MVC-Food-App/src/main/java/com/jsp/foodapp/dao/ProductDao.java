package com.jsp.foodapp.dao;

import java.util.List;

import javax.persistence.EntityManager;
import javax.persistence.EntityManagerFactory;
import javax.persistence.EntityTransaction;
import javax.persistence.Query;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Repository;

import com.jsp.foodapp.dto.Product;

@Repository
public class ProductDao {
	@Autowired
     EntityManagerFactory factory ;
	
	public void saveProduct(Product p) {
		EntityManager em = factory.createEntityManager() ;
		EntityTransaction et = em.getTransaction() ;
		
		et.begin();
		em.persist(p);
		et.commit();
	}
	
	
	public List<Product> fetchAllProducts(){
		EntityManager em = factory.createEntityManager() ;
		Query query = em.createQuery("select p from Product p") ;
		return query.getResultList() ;
	}
	
	public Product findProductById(int id) {
		EntityManager em = factory.createEntityManager() ;
		return em.find(Product.class, id) ;
	}
	
	public void updateProduct(Product product) {
		EntityManager em = factory.createEntityManager() ;
		EntityTransaction et = em.getTransaction() ;
		
		et.begin();
		em.merge(product) ;
		et.commit();
	}
	
	public void deleteProductById(int id) {
		EntityManager em = factory.createEntityManager() ;
		EntityTransaction et = em.getTransaction() ;
		Product product = em.find(Product.class, id) ;
		et.begin();
		em.remove(product) ;
		et.commit();
	}
	
}
