package com.jsp.foodapp.controller;

import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.ModelAttribute;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.servlet.ModelAndView;

import com.jsp.foodapp.dao.ProductDao;
import com.jsp.foodapp.dto.Product;
import com.jsp.foodapp.dto.User;

@Controller
public class ProductController {
	
	@Autowired
	ProductDao dao ;

	@RequestMapping("/addproduct")
	public ModelAndView addProduct() {
		ModelAndView mav = new ModelAndView("addproduct");
		Product product = new Product() ;
		mav.addObject("product", product);
		return mav;
	}

	
	@RequestMapping("/saveproduct")
	public ModelAndView saveproduct(@ModelAttribute("product") Product p) {
		ModelAndView mav = new  ModelAndView("adminoptions") ;
			dao.saveProduct(p) ;
			return mav ;
	}
	
	@RequestMapping("/viewallproducts")
	public ModelAndView viewAllProducts() {
		ModelAndView mav = new ModelAndView("allproducts") ;
		List<Product> products = dao.fetchAllProducts() ;
		mav.addObject("productlist", products) ;
		return mav ;
	}
	
	@RequestMapping("/update")
	public ModelAndView update(@RequestParam("id") int id) {
		ModelAndView mav = new ModelAndView("updateproduct") ;
		Product product =  dao.findProductById(id) ;
		mav.addObject("productinfo", product) ;
		return mav ;
	}

	@RequestMapping("/updateproduct")
	public ModelAndView updateProduct(@ModelAttribute("productinfo") Product product) {
		ModelAndView mav = new ModelAndView("adminoptions") ;
		dao.updateProduct(product) ;
		return mav ;

	}
	
	@RequestMapping("/delete")
	public ModelAndView deleteProduct(@RequestParam("id") int id) {
		ModelAndView mav = new ModelAndView("redirect:/viewallproducts") ;
	
		dao.deleteProductById(id) ;
		return mav ;
	}
	
	@RequestMapping("/viewallproduct")
	public ModelAndView viewAllProduct() {
		ModelAndView mav = new ModelAndView("displayallproducts") ;
		List<Product> products = dao.fetchAllProducts() ;
		mav.addObject("productlist", products) ;
		return mav ;
	}
}
