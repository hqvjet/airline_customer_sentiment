import React from 'react';
import { navbar } from '@/constants/var';

const Header = () => {
  return (
    <nav>
      <div className="flex justify-between  bg-slate-500 w-full text-white p-5">
        <a href='/'>MY LOGO</a>
        <div className="flex items-center justify-between gap-5">
          {navbar.map((item, index) => (
            <a key={index} href={item.href} className="hover:text-gray-300">
              {item.name}
            </a>
          ))}
        </div>
      </div>
    </nav>
  );
};

export default Header;